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
    print("Downloading/Loading pretrained ResNet18...")
    weights = models.resnet18_weights.DEFAULT
    model = models.resnet18(weights=weights)
    model = model.to(device)

    # 6. Run Evaluations
    evaluate_model(model, loader_clean, device, "Baseline (Clean Images)")
    evaluate_model(model, loader_noise1, device, "Gaussian Noise (std=1.0)")
    evaluate_model(model, loader_noise2, device, "Gaussian Noise (std=2.0)")

if __name__ == "__main__":
    main()