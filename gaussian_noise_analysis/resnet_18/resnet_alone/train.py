import os
import urllib.request
import tarfile
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import wandb
import random

os.environ['WANDB_API_KEY'] = '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'

# ==========================================
# 0. Global Constants & Fixes
# ==========================================
IMAGENETTE_CLASSES = {
    0: 'Tench', 217: 'English Springer', 482: 'Cassette Player', 
    491: 'Chain Saw', 497: 'Church', 566: 'French Horn', 
    569: 'Garbage Truck', 571: 'Gas Pump', 574: 'Golf Ball', 701: 'Parachute'
}

IMAGENETTE_TO_IMAGENET = {0:0, 1:217, 2:482, 3:491, 4:497, 5:566, 6:569, 7:571, 8:574, 9:701}

def map_class_to_imagenet(y):
    return IMAGENETTE_TO_IMAGENET[y]

def get_class_name(class_idx):
    return IMAGENETTE_CLASSES.get(class_idx, f"Class {class_idx}")

# ==========================================
# 1. Dataset Downloading & Noise Transform
# ==========================================
def download_and_extract_imagenette(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, "imagenette2-160.tgz")
    extract_path = os.path.join(data_dir, "imagenette2-160")
    
    if not os.path.exists(extract_path):
        print("Downloading Imagenette (160px version, ~160MB)...")
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
        urllib.request.urlretrieve(url, tgz_path)
        print("Extracting dataset...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        os.remove(tgz_path)
    return os.path.join(extract_path, "train"), os.path.join(extract_path, "val")

class AddGaussianNoise(object):
    def __init__(self, std=1.0):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

# ==========================================
# 2. Training & Testing Engine
# ==========================================
def train_and_evaluate(model, train_loader, clean_val_loader, noisy_val_loader, higher_order_of_noise_val_loader, device, 
                       epochs, save_interval, model_save_path, prog_vis=None, val_dataset=None, 
                       plot_every_n_epochs=1, noise_level=0.5):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_noisy_acc = 0.0
    
    print("\nStarting Training (Training ResNet-18 on CLEAN data)...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if i % 20 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
                wandb.log({"Batch Loss": loss.item()})
                
        model.eval()
        
        # Test on Clean
        clean_correct, clean_total = 0, 0
        print("Evaluating on Clean Validation Set...")
        with torch.no_grad():
            for images, labels in clean_val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images).data, 1)
                clean_total += labels.size(0)
                clean_correct += (predicted == labels).sum().item()
        clean_acc = 100 * clean_correct / clean_total
        
        # Test on Noisy
        noisy_correct, noisy_total = 0, 0
        print("Evaluating on Noisy Validation Set...")
        with torch.no_grad():
            for images, labels in noisy_val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images).data, 1)
                noisy_total += labels.size(0)
                noisy_correct += (predicted == labels).sum().item()
        noisy_acc = 100 * noisy_correct / noisy_total

        # Test on Higher Order of Noise
        higher_order_correct, higher_order_total = 0, 0
        print("Evaluating on Higher Order of Noise Validation Set...")
        with torch.no_grad():
            for images, labels in higher_order_of_noise_val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images).data, 1)
                higher_order_total += labels.size(0)
                higher_order_correct += (predicted == labels).sum().item()
        higher_order_acc = 100 * higher_order_correct / higher_order_total
        
        print(f"=== Epoch {epoch+1} | Clean Acc: {clean_acc:.2f}% | Noisy Acc: {noisy_acc:.2f}% | Higher Order Acc: {higher_order_acc:.2f}% ===")
        
        #ALWAYS check for the best model and save it 
        if noisy_acc > best_noisy_acc:
            best_noisy_acc = noisy_acc
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path)
            print(f"  [*] New best model saved at epoch {epoch+1} with Noisy Acc: {best_noisy_acc:.2f}%")
        
       
        # Log Epoch Metrics to WandB
        wandb.log({
            "Epoch": epoch + 1,
            "Clean Validation Accuracy": clean_acc,
            "Noisy Validation Accuracy": noisy_acc,
            "Higher Order Validation Accuracy": higher_order_acc,
            "Epoch Training Loss": epoch_loss / len(train_loader)
        })

        # Generate & Log Comparative Plot
        if prog_vis and val_dataset and (epoch + 1) % plot_every_n_epochs == 0:
            rand_idx = random.randint(0, len(val_dataset) - 1)
            img, true_label = val_dataset[rand_idx]
            
            img_clean = img.unsqueeze(0).to(device)
            img_noisy = img_clean + torch.randn_like(img_clean) * noise_level
            img_higher_order = img_clean + torch.randn_like(img_clean) * 2.0
            
            # Generate the plot
            fig = prog_vis.plot_comparative_progression(img_clean, img_noisy, img_higher_order, true_label, feature_map_idx=5)
            
            # Log to WandB and close the figure to avoid memory leaks
            wandb.log({f"Network Progression (Feature Map 5)": wandb.Image(fig)})
            plt.close(fig)

# ==========================================
# 3. Visualization Utilities
# ==========================================
class NetworkProgressionVisualizer:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.target_layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_children():
            if name in self.target_layers:
                def get_hook(layer_name):
                    def hook(mod, inp, out):
                        self.activations[layer_name] = out.detach()
                    return hook
                self.hooks.append(module.register_forward_hook(get_hook(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def plot_comparative_progression(self, clean_img, noisy_img, higher_order_img, true_label, feature_map_idx=0):
        with torch.no_grad():
            # Clean inference
            self.activations.clear()
            clean_pred = get_class_name(self.model(clean_img).max(1)[1].item())
            clean_acts = {k: v.clone() for k, v in self.activations.items()}
            
            # Noisy inference
            self.activations.clear()
            noisy_pred = get_class_name(self.model(noisy_img).max(1)[1].item())
            noisy_acts = {k: v.clone() for k, v in self.activations.items()}
        
            # Higher Order of Noise inference
            self.activations.clear()
            higher_order_pred = get_class_name(self.model(higher_order_img).max(1)[1].item())
            higher_order_acts = {k: v.clone() for k, v in self.activations.items()}
        
        num_cols = 1 + len(self.target_layers) 
        
        fig, axes = plt.subplots(3, num_cols, figsize=(20, 9))
        fig.suptitle(f"ResNet-18 Progression (Feature {feature_map_idx}) | True Class: {get_class_name(true_label)}", fontsize=18)
        
        def plot_inputs(axes_row, img, prefix, pred):
            axes_row[0].imshow(denormalize(img)[0].detach().cpu().permute(1, 2, 0))
            axes_row[0].set_title(f"{prefix} Input\nPred: {pred}", color='green' if prefix=="Clean" else 'black')
            axes_row[0].axis('off')

        plot_inputs(axes[0], clean_img, "Clean", clean_pred)
        plot_inputs(axes[1], noisy_img, "Noisy", noisy_pred)
        plot_inputs(axes[2], higher_order_img, "Higher Order", higher_order_pred)

        for i, layer_name in enumerate(self.target_layers):
            col = i + 1 
            if layer_name in clean_acts:
                act_c = clean_acts[layer_name][0, feature_map_idx].cpu()
                axes[0, col].imshow(act_c, cmap='viridis')
                axes[0, col].set_title(f"{layer_name}\n{act_c.shape[0]}x{act_c.shape[1]}")
            axes[0, col].axis('off')
            
            if layer_name in noisy_acts:
                act_n = noisy_acts[layer_name][0, feature_map_idx].cpu()
                axes[1, col].imshow(act_n, cmap='viridis')
                axes[1, col].set_title(f"{layer_name}\n{act_n.shape[0]}x{act_n.shape[1]}")
            axes[1, col].axis('off')
                
            if layer_name in higher_order_acts:
                act_h = higher_order_acts[layer_name][0, feature_map_idx].cpu()
                axes[2, col].imshow(act_h, cmap='viridis')
                axes[2, col].set_title(f"{layer_name}\n{act_h.shape[0]}x{act_h.shape[1]}")
            axes[2, col].axis('off')

        plt.tight_layout()
        return fig

# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize WandB 
    wandb.init(
        project="resnet18-baseline",
        config={
            "epochs": 3,
            "noise_level": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "plot_every_n_epochs": 1,
            "save_every_n_epochs": 3,
            "model_save_path": "resnet18_baseline.pth",
            "img_resize": 256,
            "img_crop": 224,
        }
    )
    
    train_transforms = transforms.Compose([
        transforms.Resize(wandb.config.img_resize), transforms.RandomCrop(wandb.config.img_crop),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([AddGaussianNoise(std=wandb.config.noise_level)], p=0.5)
    ])
    
    clean_val_transforms = transforms.Compose([
        transforms.Resize(wandb.config.img_resize), transforms.CenterCrop(wandb.config.img_crop),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    noisy_val_transforms = transforms.Compose([
        transforms.Resize(wandb.config.img_resize), transforms.CenterCrop(wandb.config.img_crop),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([AddGaussianNoise(std=wandb.config.noise_level)], p=1.0)
    ])
    higher_order_of_noise_val_transforms = transforms.Compose([
        transforms.Resize(wandb.config.img_resize), transforms.CenterCrop(wandb.config.img_crop),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([AddGaussianNoise(std=2.0)], p=1.0)
    ])
    print("Preparing Datasets...")
    train_dir, val_dir = download_and_extract_imagenette()
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms, target_transform=map_class_to_imagenet)
    clean_val_dataset = datasets.ImageFolder(val_dir, transform=clean_val_transforms, target_transform=map_class_to_imagenet)
    noisy_val_dataset = datasets.ImageFolder(val_dir, transform=noisy_val_transforms, target_transform=map_class_to_imagenet)
    higher_order_of_noise_val_dataset = datasets.ImageFolder(val_dir, transform=higher_order_of_noise_val_transforms, target_transform=map_class_to_imagenet)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=2)
    clean_val_loader = torch.utils.data.DataLoader(clean_val_dataset, batch_size=64, shuffle=False, num_workers=2)
    noisy_val_loader = torch.utils.data.DataLoader(noisy_val_dataset, batch_size=64, shuffle=False, num_workers=2)
    higher_order_of_noise_val_loader = torch.utils.data.DataLoader(higher_order_of_noise_val_dataset, batch_size=64, shuffle=False, num_workers=2)

    print("Loading ResNet-18...")
    # Initialize standard ResNet-18 (trainable by default)
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    
    # Initialize visualizer
    prog_vis = NetworkProgressionVisualizer(resnet_model)
    
    # Pass model directly to the training engine
    train_and_evaluate(
        model=resnet_model, 
        train_loader=train_loader, 
        clean_val_loader=clean_val_loader, 
        noisy_val_loader=noisy_val_loader, 
        higher_order_of_noise_val_loader=higher_order_of_noise_val_loader,
        device=device, 
        epochs=wandb.config.epochs,
        save_interval=wandb.config.save_every_n_epochs,
        model_save_path=wandb.config.model_save_path,
        prog_vis=prog_vis,
        val_dataset=clean_val_dataset,
        plot_every_n_epochs=wandb.config.plot_every_n_epochs,
        noise_level=wandb.config.noise_level
    )
    
    prog_vis.remove_hooks()
    wandb.finish()