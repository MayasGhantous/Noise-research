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

# ==========================================
# 1. Dataset Downloading & Noise Transform
# ==========================================
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
    def __init__(self, std=1.0):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


def test(model, device, test_loader, criterion):
    model.eval()
    
    higher_order_correct, higher_order_total = 0, 0
    print("Evaluating on Higher Order of Noise Validation Set...")
    with torch.no_grad():
        for images, labels in higher_order_of_noise_val_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images).data, 1)
            higher_order_total += labels.size(0)
            higher_order_correct += (predicted == labels).sum().item()
    higher_order_acc = 100 * higher_order_correct / higher_order_total
    return higher_order_acc

#test 20 rounds:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18().to(device)
    model.load_state_dict(torch.load("resnet18_baseline_noisy_train.pth", map_location=device))
    higher_order_of_noise_val_transforms = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([AddGaussianNoise(std=2.0)], p=1.0)
    ])
    train_dir, val_dir = download_and_extract_imagenette()
    higher_order_of_noise_val_dataset = datasets.ImageFolder(val_dir, transform=higher_order_of_noise_val_transforms, target_transform=map_class_to_imagenet)
    higher_order_of_noise_val_loader = torch.utils.data.DataLoader(higher_order_of_noise_val_dataset, batch_size=64, shuffle=False, num_workers=2)
    sum_test_loss = 0
    sum_accuracy = 0
    for i in range(5):
        higher_order_acc = test(model, device, higher_order_of_noise_val_loader, nn.CrossEntropyLoss())
        sum_accuracy += higher_order_acc
        print(f"Round {i+1}: Higher Order Accuracy: {higher_order_acc:.2f}%")

    print(f"Average Higher Order Accuracy: {sum_accuracy/20:.2f}%")

