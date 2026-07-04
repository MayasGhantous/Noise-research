
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from archtechre_common import *
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
            
            t_text = get_class_name(t_label)
            p_text = get_class_name(p_label)
            
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
