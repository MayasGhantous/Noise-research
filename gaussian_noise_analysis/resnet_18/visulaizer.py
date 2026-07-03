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
    