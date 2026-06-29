import os
import tarfile
import urllib.request
import requests
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# ---------------------------------------------------------
# 1. Download and Extract the Imagenette-160px Dataset
# ---------------------------------------------------------
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
dataset_path = "imagenette2-160"

if not os.path.exists(dataset_path):
    print("Downloading Imagenette 160px dataset (~95 MB)...")
    urllib.request.urlretrieve(dataset_url, "imagenette2-160.tgz")
    print("Extracting...")
    with tarfile.open("imagenette2-160.tgz", "r:gz") as tar:
        tar.extractall()
    print("Extraction complete.\n")

# ---------------------------------------------------------
# 2. Map Subset Classes to ResNet-18's 1000 Class Indices
# ---------------------------------------------------------
# Fetch the official ImageNet class index JSON
json_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
class_index = requests.get(json_url).json()

# Create a mapping from WordNet ID (folder name) to the integer class ID
# Example: "n01440764" -> 0 (tench)
wnid_to_idx = {v[0]: int(k) for k, v in class_index.items()}

# ---------------------------------------------------------
# 3. Define Image Transforms (Original vs Noisy)
# ---------------------------------------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.2):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Add noise and clamp values to stay within valid image bounds [0, 1]
        noisy = tensor + torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(noisy, 0., 1.)

# Standard ImageNet normalization values
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])

original_transform = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    normalize
])

noisy_transform = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.5), # Inject Gaussian noise
    normalize
])

# ---------------------------------------------------------
# 4. Set Up Datasets and DataLoaders
# ---------------------------------------------------------
# We only use the validation directory for testing
val_dir = os.path.join(dataset_path, "val")

orig_dataset = datasets.ImageFolder(val_dir, transform=original_transform)
noisy_dataset = datasets.ImageFolder(val_dir, transform=noisy_transform)

# Translate PyTorch ImageFolder IDs (0-9) to true ImageNet IDs (0-999)
imagenette_to_imagenet = {i: wnid_to_idx[wnid] for i, wnid in enumerate(orig_dataset.classes)}

orig_loader = DataLoader(orig_dataset, batch_size=64, shuffle=False)
noisy_loader = DataLoader(noisy_dataset, batch_size=64, shuffle=False)

# ---------------------------------------------------------
# 5. Initialize Pre-trained ResNet-18
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model with standard ImageNet weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval() # Set to evaluation mode (disables dropout, fixes batchnorm)
model.to(device)

# ---------------------------------------------------------
# 6. Evaluation Function
# ---------------------------------------------------------
def test_model(loader, description):
    correct = 0
    total = 0
    print(f"Running evaluation on {description}...")
    
    with torch.no_grad(): # No gradients needed for testing
        for images, labels in loader:
            images = images.to(device)
            
            # Map labels to the 1000-class scale for accurate comparison
            mapped_labels = torch.tensor([imagenette_to_imagenet[l.item()] for l in labels]).to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += mapped_labels.size(0)
            correct += (predicted == mapped_labels).sum().item()
            
    accuracy = 100.0 * correct / total
    return accuracy

# ---------------------------------------------------------
# 7. Run Tests
# ---------------------------------------------------------
print("-" * 40)
orig_acc = test_model(orig_loader, "Original Images")
print(f"--> Accuracy on Original Images: {orig_acc:.2f}%\n")

noisy_acc = test_model(noisy_loader, "Noisy Images (Std=0.2)")
print(f"--> Accuracy on Noisy Images: {noisy_acc:.2f}%")
print("-" * 40)