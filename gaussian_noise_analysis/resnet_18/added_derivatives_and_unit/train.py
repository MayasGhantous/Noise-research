
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import wandb
import random
from pathlib import Path
import torch.nn.functional as F


# 1. Get the absolute path to the parent directory
parent_dir = str(Path(__file__).parent.parent)

# 2. Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from orginal import *



class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetFrontEnd(nn.Module):
    """A lightweight U-Net to denoise and blend channels back into 3 channels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64) 
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)  
        
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        u1 = self.up1(x3)
        diffY = x2.size()[2] - u1.size()[2]
        diffX = x2.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x_up1 = self.conv_up1(torch.cat([x2, u1], dim=1))
        
        u2 = self.up2(x_up1)
        diffY = x1.size()[2] - u2.size()[2]
        diffX = x1.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x_up2 = self.conv_up2(torch.cat([x1, u2], dim=1))
        
        return self.outc(x_up2)

class EdgeAwareRobustifier(nn.Module):
    def __init__(self, base_resnet, blur_kernel, blur_sigma, unet_in, unet_out):
        super().__init__()
        
        self.base_resnet = base_resnet
        for param in self.base_resnet.parameters():
            param.requires_grad = False
        self.base_resnet.eval() 
            
        self.denoiser = transforms.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)
        self.unet_prep = UNetFrontEnd(in_channels=unet_in, out_channels=unet_out)
        
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 4.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3) / 4.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_derivatives(self, x):
        gray = x.mean(dim=1, keepdim=True)
        dx = F.conv2d(gray, self.sobel_x, padding=1)
        dy = F.conv2d(gray, self.sobel_y, padding=1)
        return dx, dy

    def forward(self, x):
        x_smooth = self.denoiser(x)
        dx, dy = self.compute_derivatives(x_smooth)
        x_5_channel = torch.cat([x_smooth, dx, dy], dim=1)
        x_prepared = self.unet_prep(x_5_channel)
        
        return self.base_resnet(x_prepared)
def train_model(model,config, train_loader, val_loader, val_loader2, criterion, optimizer, device,prog_vis=None):
    """
    Trains the model on the training dataset and evaluates on the validation dataset.
    """
    print("\nStarting training...")
    best_accuracy = 0.0
    
    # Watch the model to log gradients and parameters
    wandb.watch(model, criterion, log="all", log_freq=10)
    num_epochs = config.num_epochs
    plot_every_n_epochs = config.plot_every_n_epochs
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
        clean_acc = evaluate_model(model, val_loader, device, description=f"Validation after Epoch {epoch + 1}")
        noisy_acc = evaluate_model(model, val_loader2, device, description=f"Validation with Noise after Epoch {epoch + 1}")
        
        # --- Visualization Step ---
        wandb.log({
            "Epoch": epoch + 1,
            "Clean Validation Accuracy": clean_acc,
            "Noisy Validation Accuracy": noisy_acc,
            "Epoch Training Loss": epoch_loss / len(train_loader)
        })

        # Generate & Log Comparative Plot
        if prog_vis and (epoch + 1) % plot_every_n_epochs == 0:
            rand_idx = random.randint(0, len(val_loader.dataset) - 1)
            img, true_label = val_loader.dataset[rand_idx]
            
            img_clean = img.unsqueeze(0).to(device)
            img_noisy = img_clean + torch.randn_like(img_clean) * config.train_noise_std    
            img_higher_order = img_clean + torch.randn_like(img_clean) * config.eval_noise_std2
            
            # Generate the plot
            fig = prog_vis.plot_comparative_progression(img_clean, img_noisy, img_higher_order, true_label, feature_map_idx=5)
            
            # Log to WandB and close the figure to avoid memory leaks
            wandb.log({f"Network Progression (Feature Map 5)": wandb.Image(fig)})
            plt.close(fig)

        if best_accuracy <= noisy_acc:
            best_accuracy = noisy_acc
            # Save using the specific filename set in wandb config
            torch.save(model.state_dict(), wandb.config.best_model_filename)
            print(f"New best model saved as '{wandb.config.best_model_filename}' with accuracy: {best_accuracy:.2f}%")
            wandb.run.summary["best_val_accuracy_noisy"] = best_accuracy

    print("Training completed.")

# ==========================================
# 3. Visualization Utilities
# ==========================================
class NetworkProgressionVisualizer:
    def __init__(self, robust_model):
        self.model = robust_model
        self.base_resnet = robust_model.base_resnet 
        self.activations = {}
        self.hooks = []
        self.target_layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.base_resnet.named_children():
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
            self.activations.clear()
            clean_pred = get_class_name(self.model(clean_img).max(1)[1].item())
            clean_acts = {k: v.clone() for k, v in self.activations.items()}
            
            c_smooth = self.model.denoiser(clean_img)
            c_dx, c_dy = self.model.compute_derivatives(c_smooth)
            c_5ch = torch.cat([c_smooth, c_dx, c_dy], dim=1)
            c_prep = self.model.unet_prep(c_5ch) 
            
            self.activations.clear()
            noisy_pred = get_class_name(self.model(noisy_img).max(1)[1].item())
            noisy_acts = {k: v.clone() for k, v in self.activations.items()}
            
            n_smooth = self.model.denoiser(noisy_img)
            n_dx, n_dy = self.model.compute_derivatives(n_smooth)
            n_5ch = torch.cat([n_smooth, n_dx, n_dy], dim=1)
            n_prep = self.model.unet_prep(n_5ch) 

            self.activations.clear()
            higher_order_pred = get_class_name(self.model(higher_order_img).max(1)[1].item())
            higher_order_acts = {k: v.clone() for k, v in self.activations.items()}

            h_smooth = self.model.denoiser(higher_order_img)
            h_dx, h_dy = self.model.compute_derivatives(h_smooth)
            h_5ch = torch.cat([h_smooth, h_dx, h_dy], dim=1)
            h_prep = self.model.unet_prep(h_5ch)

        
        custom_stages = ['Smoothed', 'dx', 'dy', 'U-Net Out']
        num_cols = 1 + len(custom_stages) + len(self.target_layers) 
        
        fig, axes = plt.subplots(3, num_cols, figsize=(28, 9))
        fig.suptitle(f"U-Net End-to-End Progression (Feature {feature_map_idx}) | True Class: {get_class_name(true_label)}", fontsize=18)
        
        def plot_pre_resnet(axes_row, img, smooth, dx, dy, prep, prefix, pred):
            axes_row[0].imshow(denormalize(img)[0].detach().cpu().permute(1, 2, 0))
            axes_row[0].set_title(f"{prefix} Input\nPred: {pred}", color='green' if prefix=="Clean" else 'black')
            axes_row[0].axis('off')
            
            axes_row[1].imshow(denormalize(smooth)[0].detach().cpu().permute(1, 2, 0))
            axes_row[1].set_title(f"Smoothed")
            axes_row[1].axis('off')
            
            dx_disp = (dx[0, 0].cpu().numpy() + 1) / 2.0
            axes_row[2].imshow(dx_disp, cmap='gray')
            axes_row[2].set_title(f"dx (Sobel)")
            axes_row[2].axis('off')

            dy_disp = (dy[0, 0].cpu().numpy() + 1) / 2.0
            axes_row[3].imshow(dy_disp, cmap='gray')
            axes_row[3].set_title(f"dy (Sobel)")
            axes_row[3].axis('off')

            prep_idx = feature_map_idx % 3
            prep_disp = prep[0, prep_idx].cpu().numpy()
            axes_row[4].imshow(prep_disp, cmap='viridis')
            axes_row[4].set_title(f"U-Net Out\n(Ch {prep_idx})")
            axes_row[4].axis('off')

        plot_pre_resnet(axes[0], clean_img, c_smooth, c_dx, c_dy, c_prep, "Clean", clean_pred)
        plot_pre_resnet(axes[1], noisy_img, n_smooth, n_dx, n_dy, n_prep, "Noisy", noisy_pred)
        plot_pre_resnet(axes[2], higher_order_img, h_smooth, h_dx, h_dy, h_prep, "Higher-Order", higher_order_pred)
        for i, layer_name in enumerate(self.target_layers):
            col = i + 5 
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
        project="Resnet-18",
        name="Unet-kenel7",
        config={
            "learning_rate": 1e-3,
            "num_epochs": 20,
            "batch_size": 32,
            "num_workers": 2,
            "seed": 42,
            "train_split_ratio": 0.8,
            "image_resize": 256,
            "image_crop": 224,
            "train_noise_std": 1.0,
            "train_noise_prob": 0.5,
            "eval_noise_std1": 1.0,
            "eval_noise_std2": 2.0,
            "best_model_filename": "kernel7.pth",
            "plot_every_n_epochs": 1,
            "blur_kernel_size": 7,
            "blur_sigma": 1.0,
            "unet_in_channels": 5,
            "unet_out_channels": 3
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

    # 5. Load Pretrained ResNet18
    print("Downloading/Loading pretrained ResNet18...")
    base_resnet = models.resnet18()
    base_resnet.load_state_dict(torch.load("original.pth"))
    base_resnet = base_resnet.to(device)
    model = EdgeAwareRobustifier(
        base_resnet=base_resnet,
        blur_kernel=wandb.config.blur_kernel_size,
        blur_sigma=wandb.config.blur_sigma,
        unet_in=wandb.config.unet_in_channels,
        unet_out=wandb.config.unet_out_channels
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Train and finish
    train_model(model, config, train_loader, val_loader, val_loader2, criterion, optimizer, device, prog_vis=NetworkProgressionVisualizer(model))

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