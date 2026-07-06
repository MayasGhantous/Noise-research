import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)

# 2. Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from archtechre_common import *

class FFTCenterCropResize:
    def __init__(self, crop_size, output_size=(244, 244)):
        """
        Args:
            crop_size (int or tuple): The size of the center crop in the frequency domain.
                                      Smaller values mean heavier blurring/low-pass filtering.
            output_size (tuple): The final spatial dimensions to resize to.
        """
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
            
        self.output_size = output_size
        # Use bilinear interpolation for the final resize
        self.resize = transforms.Resize(output_size, interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, img_tensor):
        # 1. Store original dimensions and get FFT
        C, H, W = img_tensor.shape
        fft_result = torch.fft.fft2(img_tensor)
        
        # 2. Shift the zero-frequency (DC) component to the center
        fft_shift = torch.fft.fftshift(fft_result, dim=(-2, -1))
        
        # 3. Crop the center of the frequency spectrum
        c_h, c_w = H // 2, W // 2
        crop_h, crop_w = self.crop_size
        
        top = c_h - crop_h // 2
        bottom = top + crop_h
        left = c_w - crop_w // 2
        right = left + crop_w
        
        # Slicing the tensor to keep only the center frequencies
        cropped_fft = fft_shift[:, top:bottom, left:right]
        
        # 4. Inverse shift to move the DC component back to the corners
        ifft_shift = torch.fft.ifftshift(cropped_fft, dim=(-2, -1))
        
        # 5. Inverse FFT back to the spatial domain
        ifft_result = torch.fft.ifft2(ifft_shift)
        
        # 6. Extract the real part of the image
        img_real = torch.real(ifft_result)
        
        # Energy Scaling:
        # PyTorch's ifft2 divides by the number of elements in the NEW (cropped) tensor.
        # Since we removed frequencies, the overall magnitude is scaled up incorrectly.
        # We multiply by the ratio of the new area to the old area to restore original brightness.
        area_ratio = (crop_h * crop_w) / (H * W)
        img_real = img_real * area_ratio
        
        # Clamp values to valid image bounds just in case of ringing artifacts
        img_real = torch.clamp(img_real, 0.0, 1.0)
        
        # 7. Resize to 244x244
        final_img = self.resize(img_real)
        
        return final_img

    def __repr__(self):
        return f"{self.__class__.__name__}(crop_size={self.crop_size}, output_size={self.output_size})"

def plot_tensor_fft_channels(image_tensor):
    """
    Takes a PyTorch image tensor [C, H, W], computes its 2D Fast Fourier Transform
    for EACH channel independently, and plots the results in a grid.
    """
    # Ensure tensor is on the CPU and detached
    image_tensor = image_tensor.cpu().detach()
    
    # If a batch was passed [B, C, H, W], just grab the first image
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
        
    C, H, W = image_tensor.shape
    
    # Create a normalization copy strictly for matplotlib visualization
    # (In case your tensor has ImageNet negative values from transforms.Normalize)
    img_vis = image_tensor.clone()
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
    
    # Create a plot grid: 2 rows, (Channels + 1) columns
    fig, axes = plt.subplots(2, C + 1, figsize=(3.5 * (C + 1), 6))
    
    # --- Column 0: The Full Original Image ---
    if C == 3:
        axes[0, 0].imshow(img_vis.permute(1, 2, 0).numpy())
        axes[0, 0].set_title('Full RGB Image', fontweight='bold')
    else:
        axes[0, 0].imshow(img_vis[0].numpy(), cmap='gray')
        axes[0, 0].set_title('Original Image', fontweight='bold')
        
    axes[0, 0].axis('off')
    axes[1, 0].axis('off') # Leave bottom-left empty for layout balance
    
    channel_names = ['Red', 'Green', 'Blue'] if C == 3 else [f'Channel {i}' for i in range(C)]
    
    # --- Columns 1 to C: Independent Channels & FFTs ---
    for i in range(C):
        # Extract the single 2D grid for this channel
        channel_data = image_tensor[i]
        
        # Compute FFT
        fft_result = torch.fft.fft2(channel_data)
        fft_shift = torch.fft.fftshift(fft_result)
        magnitude_spectrum = 20 * torch.log(torch.abs(fft_shift) + 1e-8)
        
        # Row 0: Plot the raw channel intensity (in grayscale to see actual values)
        axes[0, i+1].imshow(img_vis[i].numpy(), cmap='gray')
        axes[0, i+1].set_title(f'{channel_names[i]} Intensity')
        axes[0, i+1].axis('off')
        
        # Row 1: Plot the FFT magnitude for this specific channel
        axes[1, i+1].imshow(magnitude_spectrum.numpy(), cmap='gray')
        axes[1, i+1].set_title(f'{channel_names[i]} FFT')
        axes[1, i+1].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_path, val_path = download_and_extract_imagenette()
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Example motion blur transform
        AddMotionBlur(kernel_size=41)
    ]
    
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    torch.manual_seed(42)
    np.random.seed(42)

    transform_clean = transforms.Compose([*base_transforms
                                          , normalization,])
    train_set = ImageFolder(root=train_path, transform=transform_clean, target_transform=map_class_to_imagenet)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    imges , labels = next(iter(train_loader))
    
    imge = imges[1]  # Take the first image in the batch
    plot_tensor_fft_channels(imge)
