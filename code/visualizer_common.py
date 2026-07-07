import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from archtechre_common import *

def analyze_batches(model, 
                    batch_clean_images, batch_clean_preds,
                    batch_noise1_images, batch_noise1_preds,
                    batch_noise2_images, batch_noise2_preds,device,index):
    """
    Takes 3 batches of images, their labels, and their pre-calculated predictions.
    Generates a comparison plot for every set of images at the same index.
    
    Returns a list of figures.
    """

    sample_images = []
    sample_tensors = []
    
    img_clean = batch_clean_images[index]
    img_noise1 = batch_noise1_images[index]
    img_noise2 = batch_noise2_images[index]
    
    
    sample_images.append(denormalize(img_clean.cpu()).permute(1, 2, 0).numpy())
    sample_tensors.append(img_clean.unsqueeze(0).to(device))
    
    sample_images.append(denormalize(img_noise1.cpu()).permute(1, 2, 0).numpy())
    sample_tensors.append(img_noise1.unsqueeze(0).to(device))
    
    sample_images.append(denormalize(img_noise2.cpu()).permute(1, 2, 0).numpy())
    sample_tensors.append(img_noise2.unsqueeze(0).to(device))
    
    predicted_labels = [
        get_class_name(batch_clean_preds[index].item()),
        get_class_name(batch_noise1_preds[index].item()),
        get_class_name(batch_noise2_preds[index].item())
    ]

    fig = display_multiple_images_progress(model, sample_tensors, sample_images, predicted_labels)
    
            
    return fig



def save_figures(model,visualizer,loader_clean, loader_noise1, loader_noise2,device, saving_location,max_samples=5):
    i = 0
    dictionary = {}
    
    flags = [True,False]

    for flag1 in flags:
            dictionary[flag1] = {}
            for flag2 in flags:
                dictionary[flag1][flag2] = {}
                for flag3 in flags:
                    dictionary[flag1][flag2][flag3] = {}
                    dictionary[flag1][flag2][flag3]["list"] = []
                    for labels in IMAGENETTE_CLASSES.keys():
                        #create an empty file
                        dictionary[flag1][flag2][flag3][labels] = 0

    for batch_clean, batch_noise1, batch_noise2 in tqdm.tqdm(zip(loader_clean, loader_noise1, loader_noise2)):
        # 1. Unpack batches
        images_clean, labels_clean = batch_clean
        images_noise1, labels_noise1 = batch_noise1
        images_noise2, labels_noise2 = batch_noise2

        # 2. Move images to device for batched inference
        images_clean = images_clean.to(device)
        images_noise1 = images_noise1.to(device)
        images_noise2 = images_noise2.to(device)

        # 3. Calculate predictions for the ENTIRE batch at once
        model.eval()
        with torch.no_grad():
            preds_clean = torch.argmax(model(images_clean), dim=1)
            preds_noise1 = torch.argmax(model(images_noise1), dim=1)
            preds_noise2 = torch.argmax(model(images_noise2), dim=1)

        # 4. Pass everything into your plotting function
        
        
        for j in range(len(batch_clean[0])):  # Loop through each image in the batch
            flag1= preds_clean[j] == labels_clean[j].item()
            flag2 = preds_noise1[j] == labels_noise1[j].item()
            flag3 = preds_noise2[j] == labels_noise2[j].item()
            flag1 = flag1.item()
            flag2 = flag2.item()
            flag3 = flag3.item()
            dictionary[flag1][flag2][flag3]["list"].append(i)
            if dictionary[flag1][flag2][flag3][labels_clean[j].item()] < max_samples:
                dictionary[flag1][flag2][flag3][labels_clean[j].item()] += 1
                save_path = Path(saving_location + f"/{flag1}_{flag2}_{flag3}/realLabel_{get_class_name(labels_clean[j].item())}")
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = str(save_path)
                fig = analyze_batches(
                    model,
                    images_clean, preds_clean,
                    images_noise1, preds_noise1,
                    images_noise2, preds_noise2,
                    device,j
                )
                fig.savefig(f"{save_path}/heatmap_{i}.png")
                plt.close(fig) 
                img_clean = batch_clean[0][j].squeeze(0).to(device)
                img_noisy = batch_noise1[0][j].squeeze(0).to(device)
                img_higher_order = batch_noise2[0][j].squeeze(0).to(device)
                true_label = labels_clean[j].item()

                # Generate the plot
                fig = visualizer.extract_and_return_figure(torch.stack([img_clean, img_noisy, img_higher_order]), [true_label, true_label, true_label])
                fig.savefig(f"{save_path}/feature_maps_{i}.png")
                plt.close(fig)  
            i += 1
    for flag1 in flags:
        for flag2 in flags:
            for flag3 in flags:
                with open(saving_location + f"/{flag1}_{flag2}_{flag3}/index_dictionary.json", 'w') as f:
                    json.dump(dictionary[flag1][flag2][flag3]["list"], f, indent=4)

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

        # --- ViT vs CNN Handling ---
        if len(activations.shape) == 3:
            # ViT Shape: (Batch, Sequence, Embedding_Dim) -> (B, N, D)
            B, N, D = activations.shape
            
            # Assuming Token 0 is the CLS token, extract the remaining spatial tokens
            num_spatial_tokens = N - 1
            
            # Calculate the spatial dimensions (e.g., 196 tokens -> 14x14 grid)
            h_feat = int(num_spatial_tokens ** 0.5)
            w_feat = h_feat
            
            # Strip CLS token, transpose, and reshape to mimic CNN spatial layout (B, D, H, W)
            acts = activations[:, 1:, :].transpose(1, 2).reshape(B, D, h_feat, w_feat)
            grads = gradients[:, 1:, :].transpose(1, 2).reshape(B, D, h_feat, w_feat)
            
        elif len(activations.shape) == 4:
            # CNN Shape: (Batch, Channels, Height, Width) -> already good to go
            acts = activations
            grads = gradients
            
        else:
            raise ValueError(f"Unexpected activation shape: {activations.shape}")

        # --- Standard Grad-CAM Math ---
        # Calculate weights based on the reshaped gradients
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Interpolate the small grid back to the original image size
        cam = F.interpolate(
            cam, 
            size=(input_tensor.shape[2], input_tensor.shape[3]), 
            mode='bilinear', 
            align_corners=False
        )

        # Normalize between 0 and 1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().detach().numpy()

def get_default_target_layers(model):
    """
    Helper function to automatically determine the best target layers 
    based on the model's architecture, even if it is wrapped in a UnetWrapper.
    """
    model_name = type(model).__name__
    
    # --- 1. Handle UnetWrapper ---
    if model_name == 'UNetWrapper':
        
        # We need to access the inner classification model.
        # Change 'base_model' to whatever attribute you named it inside your UnetWrapper class
        # (e.g., model.classifier, model.model, model.encoder)
        inner_model = model.get_base_model()
        model_to_inspect = inner_model
        inspect_name = type(inner_model).__name__
    else:
        # If no wrapper, we just inspect the model directly
        model_to_inspect = model
        inspect_name = model_name
        
    # --- 2. Check for ResNet (torchvision or timm) ---
    if 'ResNet' in inspect_name:
        #print(f"Detected ResNet architecture ({inspect_name}). Defaulting to Layers 1-4.")
        target_layers = [
            model_to_inspect.conv1, 
            model_to_inspect.layer1[-1], 
            model_to_inspect.layer2[-1], 
            model_to_inspect.layer3[-1], 
            model_to_inspect.layer4[-1]
        ]
        layer_names = ["Conv1", "Layer 1", "Layer 2", "Layer 3", "Layer 4"]

    # --- 3. Check for Vision Transformer ---
    elif 'VisionTransformer' in inspect_name:
        #print(f"Detected ViT architecture ({inspect_name}). Defaulting to specific Block Norms.")
        target_layers = [
            model_to_inspect.blocks[0].norm1,
            model_to_inspect.blocks[5].norm1,
            model_to_inspect.blocks[8].norm1,
            model_to_inspect.blocks[10].norm1, 
            model_to_inspect.blocks[-1].norm1
        ]
        layer_names = ["Block 0 Norm", "Block 5 Norm", "Block 8 Norm", "Block 10 Norm", "Last Block Norm"]
        
    # --- 4. Fallback: Regular CNN ---
    else:
        #sprint(f"Detected generic model ({inspect_name}). Searching for Conv2d layers...")
        # Dynamically extract all 2D Convolutional layers
        conv_layers = [m for m in model_to_inspect.modules() if isinstance(m, nn.Conv2d)]
        
        # FIX: Limit to the last 5 layers max, otherwise the plot will have too many columns and crash
        target_layers = conv_layers[-5:]
        layer_names = [f"Conv2d Layer {i+1}" for i in range(len(target_layers))]            
        
    return target_layers, layer_names

def display_multiple_images_progress(model, input_tensors, original_images, labels, target_layers=None, layer_names=None, figsize=(16, 10)):
    """
    Displays Grad-CAM progression for multiple images.
    Automatically detects model type if target_layers is not provided.
    """
    
    # --- AUTO-DETECT LAYERS IF NONE PROVIDED ---
    if target_layers is None or layer_names is None:
        target_layers, layer_names = get_default_target_layers(model)

    num_rows = len(input_tensors)
    num_cols = len(target_layers) + 2  # +2 for Original Image and Label
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Ensure axes is a 2D array even for a single image
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
        
    fig.suptitle(f"Grad-CAM ({type(model).__name__})", fontsize=18, y=0.98)
    cmap = plt.get_cmap('jet')
    
    for row in range(num_rows):
        in_tensor = input_tensors[row]
        orig_image = original_images[row]
        label = labels[row]
        
        # --- Column 0: Original Image ---
        axes[row, 0].imshow(orig_image)
        if row == 0:
            axes[row, 0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        # --- Column 1: Label ---
        axes[row, 1].text(0.5, 0.5, label, ha='center', va='center', transform=axes[row, 1].transAxes, fontsize=12)
        axes[row, 1].axis('off')

        # --- Columns 2+: Grad-CAM Layers ---
        for col, target_layer in enumerate(target_layers):
            # Generate heatmap using your existing GradCAM class
            cam_extractor = GradCAM(model, target_layer)
            heatmap = cam_extractor.generate_heatmap(in_tensor)
            
            # Apply color map and overlay
            heatmap_colored = cmap(heatmap)[..., :3]
            overlay = (0.5 * heatmap_colored) + (0.5 * orig_image)
            overlay = np.clip(overlay, 0, 1)
            
            ax = axes[row, col + 2]
            ax.imshow(overlay)
            if row == 0:
                ax.set_title(layer_names[col], fontsize=12, fontweight='bold')
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
      
class LayerFFT:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        
        self.activations = None
        self.hook_handle = self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output

    def remove_hook(self):
        """Clean up the hook after use to prevent memory leaks."""
        self.hook_handle.remove()

    def generate_fft(self, input_tensor):
        # We don't need gradients for FFT
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        activations = self.activations

        # --- ViT vs CNN Handling ---
        if len(activations.shape) == 3:
            # ViT Shape: (Batch, Sequence, Embedding_Dim) -> (B, N, D)
            B, N, D = activations.shape
            
            # Assuming Token 0 is the CLS token, extract the remaining spatial tokens
            num_spatial_tokens = N - 1
            h_feat = int(num_spatial_tokens ** 0.5)
            w_feat = h_feat
            
            # Strip CLS token, transpose, and reshape to spatial layout (B, D, H, W)
            acts = activations[:, 1:, :].transpose(1, 2).reshape(B, D, h_feat, w_feat)
            
        elif len(activations.shape) == 4:
            # CNN Shape: (Batch, Channels, Height, Width)
            acts = activations
            
        else:
            raise ValueError(f"Unexpected activation shape: {activations.shape}")

        # --- FFT Math ---
        # 1. Compute 2D FFT on the spatial dimensions of all channels
        fft_complex = torch.fft.fft2(acts)
        
        # 2. Shift the zero-frequency component to the center of the spectrum
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        
        # 3. Compute magnitude spectrum
        magnitude = torch.abs(fft_shifted)
        
        # 4. Average the magnitude across all channels in the layer
        mean_magnitude = torch.mean(magnitude, dim=1, keepdim=True)
        
        # 5. Apply Log scale to compress values and make the spectrum visible
        log_magnitude = 20 * torch.log10(mean_magnitude + 1e-8)

        # Normalize between 0 and 1 for visualization
        log_magnitude = log_magnitude - log_magnitude.min()
        log_magnitude = log_magnitude / (log_magnitude.max() + 1e-8)

        return log_magnitude.squeeze().cpu().detach().numpy()
    
def display_multiple_images_fft_progress(model, input_tensors, original_images, labels, target_layers=None, layer_names=None, figsize=(16, 10)):
    """
    Displays FFT progression of feature maps for multiple images.
    Automatically detects model type if target_layers is not provided.
    """
    
    if target_layers is None or layer_names is None:
        target_layers, layer_names = get_default_target_layers(model)

    num_rows = len(input_tensors)
    num_cols = len(target_layers) + 2  # +2 for Original Image and Label
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Ensure axes is a 2D array even for a single image
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
        
    fig.suptitle(f"Feature Map FFT Spectrum ({type(model).__name__})", fontsize=18, y=0.98)
    
    for row in range(num_rows):
        in_tensor = input_tensors[row]
        orig_image = original_images[row]
        label = labels[row]
        
        # --- Column 0: Original Image ---
        axes[row, 0].imshow(orig_image)
        if row == 0:
            axes[row, 0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        # --- Column 1: Label ---
        axes[row, 1].text(0.5, 0.5, label, ha='center', va='center', transform=axes[row, 1].transAxes, fontsize=12)
        if row == 0:
            axes[row, 1].set_title("Label", fontsize=12, fontweight='bold')
        axes[row, 1].axis('off')

        # --- Columns 2+: Layer FFTs ---
        for col, target_layer in enumerate(target_layers):
            # Generate FFT map using the LayerFFT class
            fft_extractor = LayerFFT(model, target_layer)
            fft_map = fft_extractor.generate_fft(in_tensor)
            
            # Clean up the hook
            fft_extractor.remove_hook()
            
            ax = axes[row, col + 2]
            # Use 'magma' or 'viridis' for frequency plots without overlaying on the original image
            ax.imshow(fft_map, cmap='magma')
            
            if row == 0:
                ax.set_title(layer_names[col], fontsize=12, fontweight='bold')
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig



def save_fft_map_for_an_index(model_name,group_norm,unet, index,gaussian, saving_location,load_model,models_location):
    if gaussian:
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_gaussian(batch_size=32, std1=0.5, std2=1.0)
    else:
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_motion_blur(batch_size=32, kernel_size1=20, kernel_size2=30)
    model = load_model(model_name,group_norm,unet,models_location)
    model.eval()

    sample_images = []
    sample_tensors = []
    
    img_clean = loader_clean.dataset[index][0]
    img_noise1 = loader_noise1.dataset[index][0]
    img_noise2 = loader_noise2.dataset[index][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sample_images.append(denormalize(img_clean.cpu()).permute(1, 2, 0).numpy())
    sample_tensors.append(img_clean.unsqueeze(0).to(device))
    
    sample_images.append(denormalize(img_noise1.cpu()).permute(1, 2, 0).numpy())
    sample_tensors.append(img_noise1.unsqueeze(0).to(device))
    
    sample_images.append(denormalize(img_noise2.cpu()).permute(1, 2, 0).numpy())
    sample_tensors.append(img_noise2.unsqueeze(0).to(device))
    
    predicted_labels = [
        get_class_name(loader_clean.dataset[index][1]),
        get_class_name(loader_noise1.dataset[index][1]),
        get_class_name(loader_noise2.dataset[index][1])
    ]

            
    fig = display_multiple_images_fft_progress(model,sample_tensors, sample_images, predicted_labels)
    Path(saving_location+f"/{model_name}/fft").mkdir(parents=True, exist_ok=True)
    fig.savefig(saving_location+f"/{model_name}/fft/fft_map_index_{index}.png")

