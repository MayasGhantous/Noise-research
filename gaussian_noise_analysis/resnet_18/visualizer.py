import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from PIL import Image
import torch.nn as nn
from torchvision.utils import make_grid
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)


# 2. Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from archtechre_common import *

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, 
            size=(input_tensor.shape[2], input_tensor.shape[3]), 
            mode='bilinear', 
            align_corners=False
        )

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().detach().numpy()


def display_multiple_images_progress(model, input_tensors, original_images,labels, target_layers=None,layer_names=None, figsize=(20, 12)):
    """
    Displays Grad-CAM progression for 3 separate images.
    Creates a 3x5 grid: 3 rows (one per image) and 5 columns (Original + 4 Layers).
    """
    if target_layers is None:
        target_layers = [model.layer1[0],model.layer1[1],model.layer2[0], model.layer2[1], model.layer3[0],model.layer3[1], model.layer4[0], model.layer4[1]]
    if layer_names is None:
        layer_names = ["Layer 1.0", "Layer 1.1", "Layer 2.0", "Layer 2.1", "Layer 3.0", "Layer 3.1", "Layer 4.0", "Layer 4.1"]

    # Set up a 3x5 grid figure
    num_rows = len(input_tensors)
    num_cols = len(target_layers) + 2  # +2 for the original image and label
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle("Grad-CAM Progression for 3 Images", fontsize=20, y=0.98)
    
    cmap = plt.get_cmap('jet')
    
    for row in range(num_rows):
        in_tensor = input_tensors[row]
        orig_image = original_images[row]
        label = labels[row]
        
        # --- Column 0: Original Image ---
        axes[row, 0].imshow(orig_image)
        if row == 0:
            axes[row, 0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[row, 0].axis('off')
        
        # --- Column 1: Label ---
        axes[row, 1].text(0.5, 0.5, label, ha='center', va='center', transform=axes[row, 1].transAxes)
        axes[row, 1].axis('off')

        # --- Columns 2 to 5: Grad-CAM Layers ---
        for col, target_layer in enumerate(target_layers):
            cam_extractor = GradCAM(model, target_layer)
            heatmap = cam_extractor.generate_heatmap(in_tensor)
            
            heatmap_colored = cmap(heatmap)[..., :3]
            overlay = (0.5 * heatmap_colored) + (0.5 * orig_image)
            overlay = np.clip(overlay, 0, 1)
            
            ax = axes[row, col + 2]
            ax.imshow(overlay)
            if row == 0:
                ax.set_title(layer_names[col], fontsize=14, fontweight='bold')
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_layer_kernels(layer: nn.Module, begin_idx: int, end_idx: int):
    """
    Plots the kernel weights of a specific PyTorch Conv2d layer.
    
    Args:
        layer (nn.Module): The convolutional layer to visualize.
        begin_idx (int): The starting index of the filters to plot.
        end_idx (int): The ending index of the filters to plot (exclusive).
        
    Returns:
        matplotlib.figure.Figure: The generated figure object containing the plot.
    """
    # 1. Validate that we are passing a convolutional layer
    if not isinstance(layer, nn.Conv2d):
        raise ValueError(f"Expected a Conv2d layer, but got {type(layer).__name__}")

    # 2. Extract and slice the weights 
    # The shape is [out_channels, in_channels, height, width]
    kernels = layer.weight.data.clone()
    
    # Safely constrain indices to the actual number of filters available
    begin_idx = max(0, begin_idx)
    end_idx = min(kernels.shape[0], end_idx)
    
    sliced_kernels = kernels[begin_idx:end_idx]
    
    # 3. Handle the input channels based on depth
    if sliced_kernels.shape[1] == 3:
        # Layer 1 has 3 input channels (RGB). We can visualize it directly as color.
        cmap = None 
    else:
        # Deeper layers have many input channels (e.g., 64, 128). 
        # We average across the input channel dimension (dim=1) to compress it 
        # down to a single channel (Grayscale) for visualization.
        sliced_kernels = sliced_kernels.mean(dim=1, keepdim=True)
        cmap = 'gray'

    # 4. Create the image grid
    # normalize=True and scale_each=True automatically scale the min/max 
    # of each individual filter to [0, 1] so they are visible.
    grid = make_grid(sliced_kernels, nrow=8, normalize=True, scale_each=True, padding=1)

    # 5. Convert the PyTorch tensor to a NumPy array for Matplotlib
    # Transpose from (Channels, Height, Width) to (Height, Width, Channels)
    grid_img = grid.permute(1, 2, 0).cpu().numpy()

    # 6. Plot the results and capture the figure
    fig = plt.figure(figsize=(10, 10))
    
    # If cmap='gray', matplotlib ignores it if the image has 3 channels, 
    # but we pass it anyway for the grayscale averaged matrices.
    if cmap:
        # Grayscale needs shape (H, W) or (H, W, 1) handled correctly by imshow
        plt.imshow(grid_img[:, :, 0], cmap=cmap)
    else:
        plt.imshow(grid_img)
        
    plt.axis('off')
    plt.title(f"Filters from index {begin_idx} to {end_idx - 1} \nShape visualized: {sliced_kernels.shape[1:]}")
    plt.tight_layout()
    
    # 7. Return the figure instead of calling plt.show()
    return fig

class ResNet18FeatureVisualizer:
    def __init__(self, model):
        self.model = model
        self.features = []
        self.hook_handles = []
        
        # Explicitly target the 5 major spatial stages of ResNet-18
        self.target_layers = {
            'Conv1 (112x112)': self.model.conv1,
            'Layer1 (56x56)': self.model.layer1,
            'Layer2 (28x28)': self.model.layer2,
            'Layer3 (14x14)': self.model.layer3,
            'Layer4 (7x7)': self.model.layer4
        }

    def _get_hook(self, layer_name):
        def hook(module, input, output):
            # Capture the output of the targeted residual block
            # Shape: [Batch_Size, Channels, Height, Width]
            self.features.append((layer_name, output.detach().cpu()))
        return hook

    def extract_and_return_figure(self, input_batch, true_labels, class_names=None):
        """
        Expects:
        - input_batch: shape [Batch_Size, 3, 224, 224]
        - true_labels: A tensor or list of the real labels for the batch
        - class_names (optional): A list or dict mapping label numbers to string names
        Returns a Matplotlib Figure object.
        """
        # Reset features list for a new batch
        self.features = []
        batch_size = input_batch.shape[0]
        device = next(self.model.parameters()).device
        input_batch = input_batch.to(device)

        self.model.eval()

        # 1. Register hooks to the specific ResNet stages
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._get_hook(name))
            self.hook_handles.append(handle)

        # 2. Forward pass (triggers hooks AND gets predictions)
        with torch.no_grad():
            outputs = self.model(input_batch)
            predictions = outputs.argmax(dim=1).cpu()

        # 3. Clean up all hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().tolist()
        predictions = predictions.tolist()

        # Return the figure generated by the plotting function
        return self._create_batch_figure(input_batch.cpu(), self.features, true_labels, predictions, class_names)

    def _create_batch_figure(self, input_batch, features, true_labels, predictions, class_names):
        batch_size = input_batch.shape[0]
        num_layers = len(features)  # ResNet will have 5 target layers
        num_cols = 1 + num_layers   # 1 for input image + 5 for feature maps
        
        # Create the figure
        fig, axes = plt.subplots(batch_size, num_cols, figsize=(3 * num_cols, 4 * batch_size))
        
        # Safety for single-image batches
        if batch_size == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i in range(batch_size):
            # Assumes 'denormalize' and 'get_class_name' are in your common.py scope
            vis_img = denormalize(input_batch[i]).permute(1, 2, 0).numpy()
            
            t_label = true_labels[i]
            p_label = predictions[i]
            
            # Map index to name if available
            t_text = get_class_name(t_label) if class_names is None else class_names.get(t_label, t_label)
            p_text = get_class_name(p_label) if class_names is None else class_names.get(p_label, p_label)
            
            title_color = "green" if t_label == p_label else "red"
            
            # Plot 1: Input Image
            axes[i, 0].imshow(vis_img)
            axes[i, 0].set_title(f"True: {t_text}\nPred: {p_text}", color=title_color, fontsize=14, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Plot the 5 feature maps
            for j, (layer_name, feature_tensor) in enumerate(features):
                col = j + 1
                
                # Isolate specific image's feature map: [Channels, Height, Width]
                f_map = feature_tensor[i] 
                
                # Average across all channels for overall spatial activation
                f_map_mean = f_map.mean(dim=0).numpy() 
                
                # Normalize between 0 and 1
                f_min, f_max = f_map_mean.min(), f_map_mean.max()
                if f_max - f_min > 0:
                    f_map_mean = (f_map_mean - f_min) / (f_max - f_min)
                
                # Plot
                axes[i, col].imshow(f_map_mean, cmap='viridis')
                
                # Only add titles to the top row
                if i == 0:
                    axes[i, col].set_title(layer_name, fontsize=12, fontweight='bold')
                axes[i, col].axis('off')
            
        plt.tight_layout()
        return fig

def replace_bn_with_gn(module, num_groups=32):

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Get the number of channels from the BatchNorm layer
            num_channels = child.num_features
            
            # Ensure the number of channels is divisible by the number of groups
            # If not, adjust the number of groups to 1 (which equals LayerNorm) 
            # or to the number of channels (which equals InstanceNorm).
            if num_channels % num_groups != 0:
                actual_groups = num_channels  # Fallback to InstanceNorm essentially
                print(f"Warning: {num_channels} channels not divisible by {num_groups}. "
                      f"Using {actual_groups} groups for layer '{name}'.")
            else:
                actual_groups = num_groups

            # Create the new GroupNorm layer
            gn = nn.GroupNorm(num_groups=actual_groups, num_channels=num_channels)
            
            # Replace the layer in the model
            setattr(module, name, gn)
        else:
            # Recursively apply to nested child modules
            replace_bn_with_gn(child, num_groups)
            
    return module
if __name__ == "__main__":
    # 1. Load ResNet18
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    print("Plotting first layer (RGB)...")
    plot_layer_kernels(model.conv1, begin_idx=0, end_idx=32)

    # Example 2: Visualize a deeper layer (Grayscale averaged kernels)
    # ResNet18's layer1[0].conv1 has 64 output filters, but also 64 INPUT channels.
    print("Plotting deeper layer (Grayscale averaged)...")
    plot_layer_kernels(model.layer1[0].conv1, begin_idx=0, end_idx=1)
    
    # 2. Define transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 3. Bulletproof URLs from PyTorch and Ultralytics official repos
    urls = [
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    ]
    
    input_tensors = []
    original_images = []
    
    print("Downloading images...")
    for i, url in enumerate(urls):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                img = Image.open(response).convert('RGB')
                
                # Prepare tensor for the model
                input_tensors.append(preprocess(img).unsqueeze(0))
                
                # Prepare scaled numpy array for plotting
                orig_img_np = np.array(img.resize((224, 224))) / 255.0
                original_images.append(orig_img_np)
        except Exception as e:
            print(f"Failed to download image {i+1}: {e}")
            exit()

    print("Generating Grad-CAM overlays...")
    # 4. Generate the 3-row grid plot
    fig = display_multiple_images_progress(model, input_tensors, original_images, labels=["Dog", "Zidane", "Bus"])
    
    # 5. Display the result
    plt.show()
 