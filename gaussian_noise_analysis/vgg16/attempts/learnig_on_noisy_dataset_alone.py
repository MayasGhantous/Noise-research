import sys
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Get the absolute path to the parent directory
parent_dir = str(Path(__file__).parent.parent)

# 2. Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from orginal import *

def visualize_end_to_end_progression(model, clean_dataset, noisy_dataset, device, epoch):
    """
    Creates a single, wide 2-row plot showing the end-to-end progression.
    """
    idx = torch.randint(0, len(clean_dataset), (1,)).item()
    img_clean, true_label = clean_dataset[idx]
    img_noisy, _ = noisy_dataset[idx]
    
    batch = torch.stack([img_clean, img_noisy]).to(device)

    activations = []
    layer_names = []
    hooks = []
    
    def get_hook(name):
        def hook_fn(module, input, output):
            activations.append(output.cpu().detach())
            layer_names.append(name)
        return hook_fn
        
    conv_count = 1
    for layer in model.features.children():
        if isinstance(layer, nn.Conv2d):
            layer_names.append(f"conv{conv_count}")
            hooks.append(layer.register_forward_hook(get_hook(f"conv{conv_count}")))
            conv_count += 1
    
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)
        pred_clean = preds[0].item()
        pred_noisy = preds[1].item()
        
    for handle in hooks:
        handle.remove()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    def denorm(img_tensor):
        img_vis = img_tensor.cpu().permute(1, 2, 0).numpy()
        img_vis = std * img_vis + mean
        return np.clip(img_vis, 0, 1)

    vis_clean = denorm(img_clean)
    vis_noisy = denorm(img_noisy)

    num_cols = len(activations) + 1
    fig, axes = plt.subplots(2, num_cols, figsize=(num_cols * 2.5, 6))
    fig.suptitle(f"VGG16 End-to-End Progression | True Class: {true_label}", fontsize=18, weight='bold')
    
    axes[0, 0].imshow(vis_clean)
    axes[0, 0].set_title(f"Clean Input\nPred: {pred_clean}", fontsize=10, color='green' if pred_clean == true_label else 'red')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(vis_noisy)
    axes[1, 0].set_title(f"Noisy Input\nPred: {pred_noisy}", fontsize=10, color='green' if pred_noisy == true_label else 'red')
    axes[1, 0].axis('off')
    '''
    for i in range(len(activations)):
        act_clean = activations[i][0]
        act_noisy = activations[i][1]
        name = layer_names[i]
        
        top_filter_idx = act_clean.abs().sum(dim=(1, 2)).argmax().item()
        h, w = act_clean.shape[1], act_clean.shape[2]
        
        axes[0, i+1].imshow(act_clean[top_filter_idx].numpy(), cmap='viridis')
        axes[0, i+1].set_title(f"{name}\n{h}x{w}\nCh {top_filter_idx}", fontsize=9)
        axes[0, i+1].axis('off')
        
        axes[1, i+1].imshow(act_noisy[top_filter_idx].numpy(), cmap='viridis')
        axes[1, i+1].axis('off')
    '''
    plt.tight_layout()
    wandb.log({"Network Progression": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)

def train_model(model, train_loader, val_loader, val_loader2, criterion, optimizer, device, num_epochs=5):
    """
    Trains the model on the training dataset and evaluates on the validation dataset.
    """
    print("\nStarting training...")
    best_accuracy = 0.0
    
    # Watch the model to log gradients and parameters
    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Evaluate on validation set after each epoch
        acc_clean = evaluate_model(model, val_loader, device, description=f"Validation after Epoch {epoch + 1}")
        acc_noisy = evaluate_model(model, val_loader2, device, description=f"Validation with Noise after Epoch {epoch + 1}")
        
        # --- Visualization Step ---
        visualize_end_to_end_progression(
            model=model, 
            clean_dataset=val_loader.dataset, 
            noisy_dataset=val_loader2.dataset, 
            device=device, 
            epoch=epoch+1
        )

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_accuracy,
            "val_accuracy_clean": acc_clean,
            "val_accuracy_noisy": acc_noisy
        })

        if best_accuracy < acc_noisy:
            best_accuracy = acc_noisy
            # Save using the specific filename set in wandb config
            torch.save(model.state_dict(), wandb.config.best_model_filename)
            print(f"New best model saved as '{wandb.config.best_model_filename}' with accuracy: {best_accuracy:.2f}%")
            wandb.run.summary["best_val_accuracy_noisy"] = best_accuracy

    print("Training completed.")

def train_val_split(dataset, train_indices, val_indices):
    """
    Splits a dataset into training and validation subsets.
    """
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    return train_subset, val_subset

if __name__ == "__main__":
    # --- Initialize W&B and define all constants in the config ---
    wandb.init(
        project="noisy-dataset-training",
        name="vgg16-noise-experiment",
        config={
            "learning_rate": 1e-6,
            "num_epochs": 5,
            "batch_size": 32,
            "num_workers": 2,
            "seed": 42,
            "train_split_ratio": 0.8,
            "image_resize": 256,
            "image_crop": 224,
            "train_noise_std": 1.0,
            "train_noise_prob": 0.,
            "eval_noise_std1": 1.0,
            "eval_noise_std2": 2.0,
            "best_model_filename": "best_model.pth"
            
        }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Download and Extract
    train_dir, test_dir = download_and_extract_imagenette(data_dir="./data")

    # 3. Define Image Transforms (Baseline + Noise Variations)
    base_transforms = [
        transforms.Resize(config.image_resize),
        transforms.CenterCrop(config.image_crop),
        transforms.ToTensor(),
    ]
    
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    transform_clean = transforms.Compose([*base_transforms, normalization])
    
    transform_noise_std1 = transforms.Compose([
        *base_transforms, 
        AddGaussianNoise(mean=0.0, std=config.eval_noise_std1), 
        normalization
    ])
    
    transform_noise_std2 = transforms.Compose([
        *base_transforms, 
        AddGaussianNoise(mean=0.0, std=config.eval_noise_std2), 
        normalization
    ])

    train_dataset1 = ImageFolder(root=train_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    dataset_size = len(train_dataset1)
    train_size = int(config.train_split_ratio * dataset_size)    
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    # Slice the indices into Train and Validation groups
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_transform = transforms.Compose([
        *base_transforms,
        transforms.RandomApply([AddGaussianNoise(std=config.train_noise_std)], p=config.train_noise_prob), 
        normalization
    ])


    train_dataset2 = ImageFolder(root=train_dir, transform=transform_clean, target_transform=map_class_to_imagenet)
    _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)#clean validation set
    _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)#noisey validation
    train_dataset3 = ImageFolder(root=train_dir, transform=train_transform, target_transform=map_class_to_imagenet)
    train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)#train


    train_loader = DataLoader(
        train_subset, 
        batch_size=config.batch_size, 
        shuffle=True,   
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=config.batch_size, 
        shuffle=False,  
        num_workers=config.num_workers
    )

    val_loader2 = DataLoader(
        val2_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    # 4. Load the Datasets & Loaders
    print("Loading validation datasets with different noise profiles...")
    
    dataset_clean = ImageFolder(root=test_dir, transform=transform_clean, target_transform=map_class_to_imagenet)

    loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    dataset_noise1 = ImageFolder(root=test_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    dataset_noise2 = ImageFolder(root=test_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # 5. Load Pretrained VGG16
    print("Downloading/Loading pretrained VGG16...")
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Train and finish
    train_model(model, train_loader, val_loader, val_loader2, criterion, optimizer, device, num_epochs=config.num_epochs)

    model.load_state_dict(torch.load(config.best_model_filename))
    test_acc_clean = evaluate_model(model, loader_clean, device, description="Final Test on Clean Dataset")
    test_acc_noisy1 = evaluate_model(model, loader_noise1, device, description="Final Test on Noisy Dataset (std=1.0)")
    test_acc_noisy2 = evaluate_model(model, loader_noise2, device, description="Final Test on Noisy Dataset (std=2.0)")
    wandb.run.summary["final_test_accuracy_clean"] = test_acc_clean
    wandb.run.summary["final_test_accuracy_noisy1 std=1.0"] = test_acc_noisy1
    wandb.run.summary["final_test_accuracy_noisy2 std=2.0"] = test_acc_noisy2
    # End the wandb run
    print("Training completed. Ending wandb run.")
    wandb.finish()