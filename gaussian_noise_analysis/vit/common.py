
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
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class ViTBatchAttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.qkv_output = None
        self.hook_handle = None
        # Dynamically get the number of attention heads from the model
        self.num_heads = self.model.blocks[-1].attn.num_heads

    def _get_qkv_hook(self):
        def hook(module, input, output):
            # Capture the output of the QKV linear layer 
            # Shape: [Batch_Size, Num_Tokens, 3 * Embedding_Dim]
            self.qkv_output = output.detach().cpu()
        return hook

   

    def extract_and_return_figure(self, input_batch, true_labels, class_names=None):
        batch_size = input_batch.shape[0]
        device = next(self.model.parameters()).device
        input_batch = input_batch.to(device)

        self.model.eval()

        # 1. Register the hook to the QKV projection layer instead!
        target_layer = self.model.blocks[-1].attn.qkv
        self.hook_handle = target_layer.register_forward_hook(self._get_qkv_hook())

        # 2. Forward pass
        with torch.no_grad():
            outputs = self.model(input_batch)
            predictions = outputs.argmax(dim=1).cpu()

        self.hook_handle.remove()

        # ==========================================
        # 3. MANUALLY COMPUTE THE ATTENTION MATRIX
        # ==========================================
        B, N, C = self.qkv_output.shape
        head_dim = (C // 3) // self.num_heads

        # Reshape exactly like `timm` does internally, and split into Q, K, V
        qkv = self.qkv_output.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # Each is shape [Batch, Num_Heads, Tokens, Head_Dim]

        # Calculate Attention: Softmax( (Q * K^T) / sqrt(head_dim) )
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1) # Shape: [Batch, Num_Heads, Tokens, Tokens]
        # ==========================================

        # 4. Proceed exactly as before
        cls_attention = attn[:, :, 0, 1:] 
        cls_attention = cls_attention.mean(dim=1) 
        
        grid_size = int(np.sqrt(cls_attention.shape[1]))
        cls_attention = cls_attention.reshape(batch_size, grid_size, grid_size)
        
        cls_attention = cls_attention.unsqueeze(1)
        heatmaps = F.interpolate(cls_attention, size=(input_batch.shape[2], input_batch.shape[3]), mode='bilinear', align_corners=False)
        heatmaps = heatmaps.squeeze(1).numpy() 
        
        for i in range(batch_size):
            h_min, h_max = heatmaps[i].min(), heatmaps[i].max()
            heatmaps[i] = (heatmaps[i] - h_min) / (h_max - h_min + 1e-8)

        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().tolist()
        predictions = predictions.tolist()

        return self._create_batch_figure(input_batch.cpu(), heatmaps, true_labels, predictions, class_names)

    def _create_batch_figure(self, input_batch, heatmaps, true_labels, predictions, class_names):
        batch_size = input_batch.shape[0]
        
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
        
        if batch_size == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i in range(batch_size):
            vis_img = denormalize(input_batch[i]).permute(1, 2, 0).numpy()
            heatmap = heatmaps[i]
            
            t_label = true_labels[i]
            p_label = predictions[i]
            
            t_text = class_names[t_label] if class_names else t_label
            p_text = class_names[p_label] if class_names else p_label
            
            title_color = "green" if t_label == p_label else "red"
            
            axes[i, 0].imshow(vis_img)
            axes[i, 0].set_title(f"True: {t_text} | Pred: {p_text}", color=title_color, fontsize=14, fontweight='bold')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(heatmap, cmap='jet')
            axes[i, 1].set_title("Attention Heatmap", fontsize=12)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(vis_img)
            axes[i, 2].imshow(heatmap, cmap='jet', alpha=0.5) 
            axes[i, 2].set_title("Overlay", fontsize=12)
            axes[i, 2].axis('off')
            
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