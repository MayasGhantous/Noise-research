import os
import urllib.request
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import wandb
import random

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
# 2. Advanced U-Net Front-End Architecture
# ==========================================
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

# ==========================================
# 3. Training & Testing Engine
# ==========================================
def train_and_evaluate(model, train_loader, clean_val_loader, noisy_val_loader, device, 
                       epochs, save_interval, model_save_path, prog_vis=None, val_dataset=None, 
                       plot_every_n_epochs=1, noise_level=0.5):
    
    optimizer = torch.optim.Adam(model.unet_prep.parameters(), lr=wandb.config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_noisy_acc = 0.0
    
    print("\nStarting Training (Training U-Net Front-End on CLEAN data only)...")
    for epoch in range(epochs):
        model.train()
        model.base_resnet.eval()
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
        
        print(f"=== Epoch {epoch+1} | Clean Acc: {clean_acc:.2f}% | Noisy Acc: {noisy_acc:.2f}% ===")
        
        # Save Best Model Checkpoint
        if (epoch + 1) % save_interval == 0:
            if noisy_acc > best_noisy_acc:
                best_noisy_acc = noisy_acc
                torch.save(model.state_dict(), model_save_path)
                wandb.save(model_save_path)
                print(f"  [*] Best model saved at epoch {epoch+1} with Noisy Acc: {best_noisy_acc:.2f}%")
        
        # Log Epoch Metrics to WandB
        wandb.log({
            "Epoch": epoch + 1,
            "Clean Validation Accuracy": clean_acc,
            "Noisy Validation Accuracy": noisy_acc,
            "Epoch Training Loss": epoch_loss / len(train_loader)
        })

        # Generate & Log Comparative Plot
        if prog_vis and val_dataset and (epoch + 1) % plot_every_n_epochs == 0:
            rand_idx = random.randint(0, len(val_dataset) - 1)
            img, true_label = val_dataset[rand_idx]
            
            img_clean = img.unsqueeze(0).to(device)
            img_noisy = img_clean + torch.randn_like(img_clean) * noise_level
            
            # Generate the plot
            fig = prog_vis.plot_comparative_progression(img_clean, img_noisy, true_label, feature_map_idx=5)
            
            # Log to WandB and close the figure to avoid memory leaks
            wandb.log({f"Network Progression (Feature Map 5)": wandb.Image(fig)})
            plt.close(fig)

# ==========================================
# 4. Visualization Utilities
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

    def plot_comparative_progression(self, clean_img, noisy_img, true_label, feature_map_idx=0):
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
        
        custom_stages = ['Smoothed', 'dx', 'dy', 'U-Net Out']
        num_cols = 1 + len(custom_stages) + len(self.target_layers) 
        
        fig, axes = plt.subplots(2, num_cols, figsize=(28, 6))
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
                
        plt.tight_layout()
        return fig

# ==========================================
# 5. Main Execution Block
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize WandB with expanded configuration constants
    wandb.init(
        project="unet-robustifier",
        config={
            "epochs": 20,
            "noise_level": 0.5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "plot_every_n_epochs": 1,
            "save_every_n_epochs": 3,
            "model_save_path": "kernel7_noise0,5.pth",
            "img_resize": 256,
            "img_crop": 224,
            "blur_kernel_size": 7,
            "blur_sigma": 1.0,
            "unet_in_channels": 5,
            "unet_out_channels": 3
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
    
    print("Preparing Datasets...")
    train_dir, val_dir = download_and_extract_imagenette()
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms, target_transform=map_class_to_imagenet)
    clean_val_dataset = datasets.ImageFolder(val_dir, transform=clean_val_transforms, target_transform=map_class_to_imagenet)
    noisy_val_dataset = datasets.ImageFolder(val_dir, transform=noisy_val_transforms, target_transform=map_class_to_imagenet)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=2)
    clean_val_loader = torch.utils.data.DataLoader(clean_val_dataset, batch_size=64, shuffle=False, num_workers=2)
    noisy_val_loader = torch.utils.data.DataLoader(noisy_val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print("Loading Pretrained ResNet-18...")
    base_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    
    print("Wrapping with U-Net Edge-Aware Robustifier...")
    robust_model = EdgeAwareRobustifier(
        base_resnet=base_resnet,
        blur_kernel=wandb.config.blur_kernel_size,
        blur_sigma=wandb.config.blur_sigma,
        unet_in=wandb.config.unet_in_channels,
        unet_out=wandb.config.unet_out_channels
    ).to(device)
    
    # Initialize visualizer BEFORE training
    prog_vis = NetworkProgressionVisualizer(robust_model)
    
    # Pass config parameters directly to the training engine
    train_and_evaluate(
        model=robust_model, 
        train_loader=train_loader, 
        clean_val_loader=clean_val_loader, 
        noisy_val_loader=noisy_val_loader, 
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