
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from archtechre_common import *

class ViTBatchAttentionVisualizer:
    def __init__(self, model, unet=None):
        self.model = model
        self.unet = unet
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

    def extract_and_return_figure(self, data_name, input_batch, true_labels):
        batch_size = input_batch.shape[0]
        device = next(self.model.parameters()).device
        input_batch = input_batch.to(device)

        # 1. Forward pass through U-Net if provided
        if self.unet is not None:
            self.unet.eval()
            self.unet = self.unet.to(device)
            with torch.no_grad():
                unet_batch = self.unet(input_batch)
        else:
            unet_batch = input_batch

        self.model.eval()

        # 2. Register the hook to the QKV projection layer
        target_layer = self.model.blocks[-1].attn.qkv
        self.hook_handle = target_layer.register_forward_hook(self._get_qkv_hook())

        # 3. Forward pass through ViT
        with torch.no_grad():
            outputs = self.model(unet_batch) # Pass the U-Net output to the ViT (or original if unet is None)
            predictions = outputs.argmax(dim=1).cpu()

        self.hook_handle.remove()

        # ==========================================
        # 4. MANUALLY COMPUTE THE ATTENTION MATRIX
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

        # 5. Process Heatmaps
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

        return self._create_batch_figure(
            data_name,
            input_batch.cpu(), 
            unet_batch.detach().cpu(), 
            heatmaps, 
            true_labels, 
            predictions
            
        )
    def _create_batch_figure(self,data_name, input_batch, unet_batch, heatmaps, true_labels, predictions):
        batch_size = input_batch.shape[0]
        has_unet = self.unet is not None
        num_cols = 4 if has_unet else 3
        
        # Adjust figure size dynamically based on the number of columns
        fig, axes = plt.subplots(batch_size, num_cols, figsize=(5 * num_cols, 5 * batch_size))
        
        if batch_size == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i in range(batch_size):
            # Assuming denormalize and get_class_name are defined in your outer scope
            vis_img = denormalize(input_batch[i]).permute(1, 2, 0).numpy()
            heatmap = heatmaps[i]
            
            t_label = true_labels[i]
            p_label = predictions[i]
            
            t_text = get_class_name(data_name=data_name, class_idx=t_label)
            p_text = get_class_name(data_name=data_name, class_idx=p_label)
            
            title_color = "green" if t_label == p_label else "red"
            
            col_idx = 0
            
            # Column 1: Original Image
            axes[i, col_idx].imshow(vis_img)
            if has_unet:
                axes[i, col_idx].set_title(f"Original Input\nTrue: {t_text}", fontsize=14)
            else:
                axes[i, col_idx].set_title(f"True: {t_text} | Pred: {p_text}", color=title_color, fontsize=14, fontweight='bold')
            axes[i, col_idx].axis('off')
            col_idx += 1
            
            # Column 2: U-Net Preprocessor Output (Conditional)
            if has_unet:
                u_img = unet_batch[i]
                u_min, u_max = u_img.min(), u_img.max()
                unet_vis_img = ((u_img - u_min) / (u_max - u_min + 1e-8)).permute(1, 2, 0).numpy()

                axes[i, col_idx].imshow(unet_vis_img)
                axes[i, col_idx].set_title(f"U-Net Feature Map\nPred: {p_text}", color=title_color, fontsize=14, fontweight='bold')
                axes[i, col_idx].axis('off')
                col_idx += 1

            # Column 3: Attention Heatmap
            axes[i, col_idx].imshow(heatmap, cmap='jet')
            axes[i, col_idx].set_title("Attention Heatmap", fontsize=14)
            axes[i, col_idx].axis('off')
            col_idx += 1
            
            # Column 4: Overlay on Original Image
            axes[i, col_idx].imshow(vis_img)
            axes[i, col_idx].imshow(heatmap, cmap='jet', alpha=0.5) 
            axes[i, col_idx].set_title("Overlay", fontsize=14)
            axes[i, col_idx].axis('off')
            
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
