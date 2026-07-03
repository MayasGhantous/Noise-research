
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
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import urllib.request
import tarfile
from torch.utils.data import Subset
import timm


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
    """
    Custom transform to add Gaussian noise to a tensor.
    """
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Generates noise with the same shape as the input tensor
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)



def evaluate_model(model, dataloader, device, description=""):
    """
    Runs the validation loop for a given model and dataloader.
    """
    print(f"\nStarting evaluation: {description}...")
    correct = 0
    total = 0

    # Ensure model is in eval mode
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("-" * 50)
    print(f"[{description}] Accuracy: {accuracy:.2f}%")
    print("-" * 50)
    
    return accuracy

def train_val_split(dataset, train_indices, val_indices):
    """
    Splits a dataset into training and validation subsets.
    """
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    return train_subset, val_subset

import torch
import matplotlib.pyplot as plt
import math

class ViTProgressionVisualizer:
    def __init__(self, model, target_layers=None):
        self.model = model
        self.activations = {}
        self.hooks = []
        # Defaults updated to standard ViT block names
        self.target_layers = target_layers or ['blocks.0', 'blocks.3', 'blocks.7', 'blocks.11']
        self._register_hooks()

    def _register_hooks(self):
        # Changed to named_modules() to easily find nested layers like 'blocks.0'
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                def get_hook(layer_name):
                    def hook(mod, inp, out):
                        # Some implementations return tuples, we just need the tensor
                        if isinstance(out, tuple):
                            out = out[0]
                        self.activations[layer_name] = out.detach()
                    return hook
                self.hooks.append(module.register_forward_hook(get_hook(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def _format_vit_activation(self, act_tensor, feature_map_idx):
        """Converts ViT token sequence [Batch, Tokens, Dim] into a 2D spatial grid."""
        act = act_tensor[0] # Grab the first item in the batch
        
        # Check if it's a 2D token sequence (Tokens x Embed_Dim)
        if act.dim() == 2: 
            # 1. Remove the classification token (usually the 0th token)
            act = act[1:, :] 
            
            # 2. Calculate the grid size (e.g., sqrt of 196 is 14)
            grid_size = int(math.sqrt(act.shape[0]))
            embed_dim = act.shape[1]
            
            # 3. Reshape into [Height, Width, Channels]
            act = act.reshape(grid_size, grid_size, embed_dim)
            
            # 4. Reorder to [Channels, Height, Width] so we can select by index
            act = act.permute(2, 0, 1) 
            
        return act[feature_map_idx].cpu()

    def plot_comparative_progression(self, clean_img, noisy_img, higher_order_img, true_label, feature_map_idx=0, get_class_name=None, denormalize=None):
        
        # Ensure helper functions are defined or passed in
        get_class_name = get_class_name or (lambda x: str(x))
        denormalize = denormalize or (lambda x: x)

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
        fig.suptitle(f"ViT-Tiny Progression (Feature {feature_map_idx}) | True Class: {get_class_name(true_label)}", fontsize=18)
        
        def plot_inputs(axes_row, img, prefix, pred):
            # Assumes img is shape [1, C, H, W]
            img_to_plot = denormalize(img)[0].detach().cpu().permute(1, 2, 0)
            img_to_plot = torch.clamp(img_to_plot, 0, 1) # Prevents matplotlib warnings
            axes_row[0].imshow(img_to_plot)
            axes_row[0].set_title(f"{prefix} Input\nPred: {pred}", color='green' if prefix=="Clean" else 'black')
            axes_row[0].axis('off')

        plot_inputs(axes[0], clean_img, "Clean", clean_pred)
        plot_inputs(axes[1], noisy_img, "Noisy", noisy_pred)
        plot_inputs(axes[2], higher_order_img, "Higher Order", higher_order_pred)

        for i, layer_name in enumerate(self.target_layers):
            col = i + 1 
            
            if layer_name in clean_acts:
                act_c = self._format_vit_activation(clean_acts[layer_name], feature_map_idx)
                axes[0, col].imshow(act_c, cmap='viridis')
                axes[0, col].set_title(f"{layer_name}\n{act_c.shape[0]}x{act_c.shape[1]}")
            axes[0, col].axis('off')
            
            if layer_name in noisy_acts:
                act_n = self._format_vit_activation(noisy_acts[layer_name], feature_map_idx)
                axes[1, col].imshow(act_n, cmap='viridis')
                axes[1, col].set_title(f"{layer_name}\n{act_n.shape[0]}x{act_n.shape[1]}")
            axes[1, col].axis('off')
                
            if layer_name in higher_order_acts:
                act_h = self._format_vit_activation(higher_order_acts[layer_name], feature_map_idx)
                axes[2, col].imshow(act_h, cmap='viridis')
                axes[2, col].set_title(f"{layer_name}\n{act_h.shape[0]}x{act_h.shape[1]}")
            axes[2, col].axis('off')

        plt.tight_layout()
        return fig
    
class ViTGroupNormWrapper(nn.Module):
    """
    A wrapper that permutes ViT dimensions to make them compatible 
    with PyTorch's native GroupNorm computation.
    """
    def __init__(self, num_channels, num_groups=1, eps=1e-5):
        super().__init__()
        # GroupNorm requires (num_groups, num_channels)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)

    def forward(self, x):
        # ViT input shape: [Batch, Sequence_Length, Channels]
        # We must permute to: [Batch, Channels, Sequence_Length] for GroupNorm
        x = x.permute(0, 2, 1)
        
        # Apply GroupNorm
        x = self.gn(x)
        
        # Permute back to original ViT shape: [Batch, Sequence_Length, Channels]
        x = x.permute(0, 2, 1)
        return x

def replace_vit_layernorm_with_groupnorm(module, num_groups=1):
    """
    Recursively searches a model for nn.LayerNorm and replaces it with 
    our custom ViTGroupNormWrapper.
    """
    for name, child in module.named_children():
        # If the child is a LayerNorm, replace it
        if isinstance(child, nn.LayerNorm):
            
            # Extract the channel count from the existing LayerNorm
            if isinstance(child.normalized_shape, int):
                num_channels = child.normalized_shape
            else:
                num_channels = child.normalized_shape[0]
            
            # Initialize our wrapper with the correct channels and desired groups
            replacement = ViTGroupNormWrapper(
                num_channels=num_channels, 
                num_groups=num_groups, 
                eps=child.eps
            )
            
            # Overwrite the LayerNorm layer in the parent module
            setattr(module, name, replacement)
            
        else:
            # If it's a block or sequential container, recurse deeper
            replace_vit_layernorm_with_groupnorm(child, num_groups)
            
    return module

def main():
    # 1. Configuration & Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Download and Extract
    train_dir, val_dir = download_and_extract_imagenette(data_dir="./data")

    # 3. Define Image Transforms (Baseline + Noise Variations)
    # We apply noise AFTER converting to tensor but BEFORE normalization, 
    # though applying it after normalization is also a valid testing strategy.
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    transform_clean = transforms.Compose([*base_transforms, normalization])
    
    transform_noise_std1 = transforms.Compose([
        *base_transforms, 
        AddGaussianNoise(mean=0.0, std=1.0), 
        normalization
    ])
    
    transform_noise_std2 = transforms.Compose([
        *base_transforms, 
        AddGaussianNoise(mean=0.0, std=2.0), 
        normalization
    ])

    # 4. Load the Datasets & Loaders
    print("Loading validation datasets with different noise profiles...")
    
    dataset_clean = ImageFolder(root=val_dir, transform=transform_clean, target_transform=map_class_to_imagenet)
    loader_clean = DataLoader(dataset_clean, batch_size=32, shuffle=False, num_workers=2)

    dataset_noise1 = ImageFolder(root=val_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=32, shuffle=False, num_workers=2)

    dataset_noise2 = ImageFolder(root=val_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=32, shuffle=False, num_workers=2)

    # 5. Load Pretrained ResNet18
    

    # 3. Load the model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
    model = replace_vit_layernorm_with_groupnorm(model, num_groups=16).to(device) 

    # 6. Run Evaluations
    evaluate_model(model, loader_clean, device, "Baseline (Clean Images)")
    evaluate_model(model, loader_noise1, device, "Gaussian Noise (std=1.0)")
    evaluate_model(model, loader_noise2, device, "Gaussian Noise (std=2.0)")

if __name__ == "__main__":
    main()