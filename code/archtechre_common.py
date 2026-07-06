import csv
from pathlib import Path
import shutil
import zipfile
import cv2
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import os
import urllib.request
import tarfile
import random
import wandb
import matplotlib.pyplot as plt
import numpy as np
import tqdm 
import torch.nn.functional as F
import torch.nn as nn

if not os.environ.get("SCRIPT_ALREADY_RAN"):
    print("setting seeds")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.environ["SCRIPT_ALREADY_RAN"] = "True"

def download_and_extract_imagenette(data_dir="data"):
    #get the absolute path to the parent directory
    parent_dir = str(Path(__file__).parent.parent)
    data_dir = os.path.join(parent_dir, data_dir)
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
    else:
        print("Dataset already exists locally.")
        
    return os.path.join(extract_path, "train"), os.path.join(extract_path, "val")


def prepare_gtsrb_for_imagefolder(data_dir="data"):
    # Get the absolute path to the parent directory
    parent_dir = str(Path(__file__).parent.parent)
    data_dir = os.path.join(parent_dir, data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    gtsrb_root = os.path.join(data_dir, "GTSRB")
    
    # --- 1. DOWNLOAD & EXTRACT TRAINING DATA ---
    train_zip = os.path.join(data_dir, "GTSRB_Final_Training_Images.zip")
    train_path = os.path.join(gtsrb_root, "Final_Training", "Images")
    
    if not os.path.exists(train_path):
        print("Downloading Train Data...")
        url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        urllib.request.urlretrieve(url, train_zip)
        print("Extracting Train Data...")
        with zipfile.ZipFile(train_zip, "r") as z:
            z.extractall(data_dir)
        os.remove(train_zip)
        
    # --- 2. DOWNLOAD & EXTRACT RAW TESTING DATA ---
    test_img_zip = os.path.join(data_dir, "GTSRB_Final_Test_Images.zip")
    test_gt_zip = os.path.join(data_dir, "GTSRB_Final_Test_GT.zip")
    raw_test_path = os.path.join(gtsrb_root, "Final_Test", "Images")
    
    # Paths for where the CSV *should* go
    test_csv_dir = os.path.join(gtsrb_root, "Final_Test")
    test_csv_path = os.path.join(test_csv_dir, "GT-final_test.csv")
    
    if not os.path.exists(raw_test_path):
        print("Downloading Test Images...")
        url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
        urllib.request.urlretrieve(url, test_img_zip)
        print("Extracting Test Images...")
        with zipfile.ZipFile(test_img_zip, "r") as z:
            z.extractall(data_dir)
        os.remove(test_img_zip)
        
    if not os.path.exists(test_csv_path):
        print("Downloading Test Ground Truth CSV...")
        url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
        urllib.request.urlretrieve(url, test_gt_zip)
        
        # FIX: Ensure the Final_Test directory exists, and extract the CSV specifically into it
        os.makedirs(test_csv_dir, exist_ok=True)
        with zipfile.ZipFile(test_gt_zip, "r") as z:
            z.extractall(test_csv_dir) 
        os.remove(test_gt_zip)

    # --- 3. FORMAT TESTING DATA FOR IMAGEFOLDER ---
    formatted_test_path = os.path.join(gtsrb_root, "Test_ImageFolder")
    
    if not os.path.exists(formatted_test_path):
        print("Organizing test images into class folders for ImageFolder...")
        os.makedirs(formatted_test_path)
        
        with open(test_csv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                filename = row['Filename']
                class_id = row['ClassId']
                
                class_folder = f"{int(class_id):05d}"
                
                src_path = os.path.join(raw_test_path, filename)
                dst_dir = os.path.join(formatted_test_path, class_folder)
                dst_path = os.path.join(dst_dir, filename)
                
                os.makedirs(dst_dir, exist_ok=True)
                
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                    
        print("Test data formatted successfully.")
    else:
        print("Dataset already downloaded and formatted.")

    return train_path, formatted_test_path

# --- Class Mapping Dictionaries & Functions ---
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

# --- Custom Noise Transform ---
class AddMotionBlur(object):
    """
    Ultra-fast custom transform to add motion blur using OpenCV precomputed kernels.
    """
    def __init__(self, kernel_size=15, angle_range=(0.0, 360.0), num_angles=360):
        """
        Args:
            kernel_size (int): The intensity/length of the blur. Must be an odd number.
            angle_range (tuple or float): A tuple (min_angle, max_angle) for random angles.
            num_angles (int): How many distinct angles to precompute.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.kernel_size = kernel_size
        self.kernels = []
        
        # --- PRECOMPUTE OPENCV KERNELS ONCE ---
        if isinstance(angle_range, (tuple, list)):
            min_angle, max_angle = angle_range
            step = (max_angle - min_angle) / max(1, num_angles - 1)
            angles_to_compute = [min_angle + i * step for i in range(num_angles)]
        else:
            angles_to_compute = [float(angle_range)]
            
        for angle in angles_to_compute:
            # Create a blank numpy kernel
            kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
            
            # Draw the straight line
            kernel[self.kernel_size // 2, :] = 1.0
            
            # Rotate it using OpenCV's highly optimized warpAffine
            center = (self.kernel_size / 2 - 0.5, self.kernel_size / 2 - 0.5)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation_matrix, (self.kernel_size, self.kernel_size))
            
            # Normalize
            kernel = kernel / np.sum(kernel)
            self.kernels.append(kernel)

    def __call__(self, tensor):
        # 1. Grab a random precomputed kernel
        idx = random.randint(0, len(self.kernels) - 1)
        kernel = self.kernels[idx]
        
        # 2. Convert PyTorch Tensor [C, H, W] -> Numpy Array [H, W, C]
        img_np = tensor.permute(1, 2, 0).numpy()
        
        # 3. Apply OpenCV's lightning-fast filter2D (Applies to all channels at once)
        blurred_np = cv2.filter2D(img_np, -1, kernel)
        
        # 4. Convert back to PyTorch Tensor [H, W, C] -> [C, H, W]
        blurred_tensor = torch.from_numpy(blurred_np).permute(2, 0, 1)
        
        return blurred_tensor
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

# --- Evaluation Function ---
def evaluate_model(model, dataloader, device, description=""):
    """
    Runs the validation loop for a given model and dataloader.
    """
    correct = 0
    total = 0

    # Ensure model is in eval mode
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader, desc=f"Processing {description}"):
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


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


def get_test_loaders_for_gaussian(batch_size=32,std1=0.5,std2=1,data_name = "imagenette"):

    if data_name == "imagenette":
        _, val_dir = download_and_extract_imagenette()
    elif data_name == "gtsrb":
        _, val_dir = prepare_gtsrb_for_imagefolder()

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
        AddGaussianNoise(mean=0.0, std=std1), 
        normalization
    ])
    
    transform_noise_std2 = transforms.Compose([
        *base_transforms, 
        AddGaussianNoise(mean=0.0, std=std2), 
        normalization
    ])

    # 4. Load the Datasets & Loaders
    print("Loading validation datasets with different noise profiles...")
    
    dataset_clean = ImageFolder(root=val_dir, transform=transform_clean, target_transform=map_class_to_imagenet)
    loader_clean = DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=2)

    dataset_noise1 = ImageFolder(root=val_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=batch_size, shuffle=False, num_workers=2)

    dataset_noise2 = ImageFolder(root=val_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=batch_size, shuffle=False, num_workers=2)

    return loader_clean, loader_noise1, loader_noise2



def get_test_loaders_for_motion_blur(batch_size=32,kernel_size1=15,kernel_size2=25,data_name = "imagenette"):

    if data_name == "imagenette":
        _, val_dir = download_and_extract_imagenette()
    elif data_name == "gtsrb":
        _, val_dir = prepare_gtsrb_for_imagefolder()

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
        AddMotionBlur(kernel_size=kernel_size1),
        normalization
    ])
    
    transform_noise_std2 = transforms.Compose([
        *base_transforms, 
        AddMotionBlur(kernel_size=kernel_size2),
        normalization
    ])

    # 4. Load the Datasets & Loaders
    print("Loading validation datasets with different noise profiles...")
    if data_name == "imagenette":
            
        dataset_clean = ImageFolder(root=val_dir, transform=transform_clean, target_transform=map_class_to_imagenet)
        loader_clean = DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=2)

        dataset_noise1 = ImageFolder(root=val_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
        loader_noise1 = DataLoader(dataset_noise1, batch_size=batch_size, shuffle=False, num_workers=2)

        dataset_noise2 = ImageFolder(root=val_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
        loader_noise2 = DataLoader(dataset_noise2, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        dataset_clean = ImageFolder(root=val_dir, transform=transform_clean)
        loader_clean = DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=2)

        dataset_noise1 = ImageFolder(root=val_dir, transform=transform_noise_std1)
        loader_noise1 = DataLoader(dataset_noise1, batch_size=batch_size, shuffle=False, num_workers=2)

        dataset_noise2 = ImageFolder(root=val_dir, transform=transform_noise_std2)
        loader_noise2 = DataLoader(dataset_noise2, batch_size=batch_size, shuffle=False, num_workers=2)

    return loader_clean, loader_noise1, loader_noise2




def train_val_split(dataset, train_indices, val_indices):
    """
    Splits a dataset into training and validation subsets.
    """
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    return train_subset, val_subset


def get_traing_val_test_loaders_for_gaussian(config):
    if config.data_name == "imagenette":
        train_dir, test_dir = download_and_extract_imagenette()
    elif config.data_name == "gtsrb":
        train_dir, test_dir = prepare_gtsrb_for_imagefolder()

    # 3. Define Image Transforms (Baseline + Noise Variations)
    base_transforms = [
        transforms.Resize(config.image_resize),
        transforms.CenterCrop(config.image_crop),
        transforms.ToTensor(),
    ]
    
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    transform_clean = transforms.Compose([*base_transforms, normalization])
    
    transform_noise_std1 = transforms.Compose([
        *base_transforms, 
        AddGaussianNoise(mean=0.0, std=config.eval_noise_std1), 
        normalization
    ])
    
    transform_noise_std2 = transforms.Compose([
        *base_transforms, 
        AddGaussianNoise(mean=0.0, std=config.eval_noise_std2), 
        normalization
    ])
    if config.data_name == "imagenette":
        train_dataset1 = ImageFolder(root=train_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    else:
        train_dataset1 = ImageFolder(root=train_dir, transform=transform_noise_std1)
                                     
    dataset_size = len(train_dataset1)
    train_size = int(config.train_split_ratio * dataset_size)    
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    # Slice the indices into Train and Validation groups
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_transform = transforms.Compose([
        *base_transforms,
        transforms.RandomApply([AddGaussianNoise(std=config.train_noise_std)], p=config.train_noise_prob), 
        normalization
    ])

    if config.data_name == "imagenette":
        train_dataset2 = ImageFolder(root=train_dir, transform=transform_clean, target_transform=map_class_to_imagenet)
        _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)#clean validation set
        _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)#noisey validation
        train_dataset3 = ImageFolder(root=train_dir, transform=train_transform, target_transform=map_class_to_imagenet)
        train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)#train

        train_dataset4 = ImageFolder(root=train_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
        _, val3_subset = train_val_split(train_dataset4, train_indices, val_indices)
    else:
        train_dataset2 = ImageFolder(root=train_dir, transform=transform_clean)
        _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)#clean validation set
        _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)#noisey validation
        train_dataset3 = ImageFolder(root=train_dir, transform=train_transform)
        train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)#train

        train_dataset4 = ImageFolder(root=train_dir, transform=transform_noise_std2)
        _, val3_subset = train_val_split(train_dataset4, train_indices, val_indices)

    train_loader = DataLoader(
        train_subset, 
        batch_size=config.batch_size, 
        shuffle=True,   
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=config.batch_size, 
        shuffle=False,  
        num_workers=config.num_workers
    )

    val_loader2 = DataLoader(
        val2_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )

    val_loader3 = DataLoader(
        val3_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    # 4. Load the Datasets & Loaders
    print("Loading validation datasets with different noise profiles...")

    if config.data_name == "imagenette":

        dataset_clean = ImageFolder(root=test_dir, transform=transform_clean, target_transform=map_class_to_imagenet)

        loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        dataset_noise1 = ImageFolder(root=test_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
        loader_noise1 = DataLoader(dataset_noise1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        dataset_noise2 = ImageFolder(root=test_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
        loader_noise2 = DataLoader(dataset_noise2, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    else:
        dataset_clean = ImageFolder(root=test_dir, transform=transform_clean)
        loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        dataset_noise1 = ImageFolder(root=test_dir, transform=transform_noise_std1)
        loader_noise1 = DataLoader(dataset_noise1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        dataset_noise2 = ImageFolder(root=test_dir, transform=transform_noise_std2)
        loader_noise2 = DataLoader(dataset_noise2, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2


def get_traing_val_test_loaders_for_motion_blure(config):
    if config.data_name == "imagenette": 
        train_dir, test_dir = download_and_extract_imagenette()
    elif config.data_name == "gtsrb":
        train_dir, test_dir = prepare_gtsrb_for_imagefolder()

    # 3. Define Image Transforms (Baseline + Noise Variations)
    base_transforms = [
        transforms.Resize(config.image_resize),
        transforms.CenterCrop(config.image_crop),
        transforms.ToTensor(),
    ]
    
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    transform_clean = transforms.Compose([*base_transforms, normalization])
    
    transform_noise_std1 = transforms.Compose([
        *base_transforms, 
        AddMotionBlur(kernel_size=config.kernel_size1), 
        normalization
    ])
    
    transform_noise_std2 = transforms.Compose([
        *base_transforms, 
        AddMotionBlur(kernel_size=config.kernel_size2), 
        normalization
    ])
    if config.data_name == "imagenette":
        train_dataset1 = ImageFolder(root=train_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    else:
        train_dataset1 = ImageFolder(root=train_dir, transform=transform_noise_std1)
    dataset_size = len(train_dataset1)
    train_size = int(config.train_split_ratio * dataset_size)    
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    # Slice the indices into Train and Validation groups
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_transform = transforms.Compose([
        *base_transforms,
        transforms.RandomApply([AddMotionBlur(kernel_size=config.kernel_size1)], p=config.train_noise_prob), 
        normalization
    ])

    if config.data_name == "imagenette":
        train_dataset2 = ImageFolder(root=train_dir, transform=transform_clean, target_transform=map_class_to_imagenet)
        _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)#clean validation set
        _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)#noisey validation
        train_dataset3 = ImageFolder(root=train_dir, transform=train_transform, target_transform=map_class_to_imagenet)
        train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)#train

        train_dataset4 = ImageFolder(root=train_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
        _, val3_subset = train_val_split(train_dataset4, train_indices, val_indices)#
    else:

        train_dataset2 = ImageFolder(root=train_dir, transform=transform_clean)
        _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)#clean validation set
        _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)#noisey validation
        train_dataset3 = ImageFolder(root=train_dir, transform=train_transform)
        train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)#train

        train_dataset4 = ImageFolder(root=train_dir, transform=transform_noise_std2)
        _, val3_subset = train_val_split(train_dataset4, train_indices, val_indices)#

    train_loader = DataLoader(
        train_subset, 
        batch_size=config.batch_size, 
        shuffle=True,   
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=config.batch_size, 
        shuffle=False,  
        num_workers=config.num_workers
    )

    val_loader2 = DataLoader(
        val2_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )

    val_loader3 = DataLoader(
        val3_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    # 4. Load the Datasets & Loaders
    print("Loading validation datasets with different noise profiles...")

    if config.data_name == "imagenette":
        dataset_clean = ImageFolder(root=test_dir, transform=transform_clean, target_transform=map_class_to_imagenet)

        loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        dataset_noise1 = ImageFolder(root=test_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
        loader_noise1 = DataLoader(dataset_noise1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        dataset_noise2 = ImageFolder(root=test_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
        loader_noise2 = DataLoader(dataset_noise2, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    else:
        dataset_clean = ImageFolder(root=test_dir, transform=transform_clean)
        loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        dataset_noise1 = ImageFolder(root=test_dir, transform=transform_noise_std1)
        loader_noise1 = DataLoader(dataset_noise1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        dataset_noise2 = ImageFolder(root=test_dir, transform=transform_noise_std2)
        loader_noise2 = DataLoader(dataset_noise2, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2


def train_model(model, train_loader, val_loader, val_loader2,val_loader3, criterion, optimizer, device,prog_vis=None,config=None):
    """
    Trains the model on the training dataset and evaluates on the validation dataset.
    """
    print("\nStarting training...")
    best_accuracy = 0.0
    num_epochs = config.num_epochs
    plot_every_n_epochs = config.plot_every_n_epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Evaluate on validation set after each epoch
        clean_acc = evaluate_model(model, val_loader, device, description=f"Validation after Epoch {epoch + 1}")
        noisy_acc = evaluate_model(model, val_loader2, device, description=f"Validation with Noise after Epoch {epoch + 1}")
        higher_order_acc = evaluate_model(model, val_loader3, device, description=f"Validation with Higher Order Noise after Epoch {epoch + 1}")
        
        # --- Visualization Step ---
        wandb.log({
            "Epoch_accuracy": epoch_accuracy,
            "Clean Validation Accuracy": clean_acc,
            "Noisy Validation Accuracy": noisy_acc,
            "Higher Order Validation Accuracy": higher_order_acc,
            "Epoch Training Loss": epoch_loss / len(train_loader)
        })

        # Generate & Log Comparative Plot
        if prog_vis and (epoch + 1) % plot_every_n_epochs == 0:
            rand_idx = random.randint(0, len(val_loader.dataset) - 1)
            img, true_label = val_loader.dataset[rand_idx]
            img2, _ = val_loader2.dataset[rand_idx]
            img3, _ = val_loader3.dataset[rand_idx]
            
            img_clean = img.squeeze(0).to(device)
            img_noisy = img2.squeeze(0).to(device)
            img_higher_order = img3.squeeze(0).to(device)
            
            # Generate the plot
            fig = prog_vis.extract_and_return_figure(torch.stack([img_clean, img_noisy, img_higher_order]), [true_label, true_label, true_label])
            
            # Log to WandB and close the figure to avoid memory leaks
            wandb.log({f"Network Progression: ": wandb.Image(fig)})
            plt.close(fig)

        if best_accuracy <= noisy_acc:
            best_accuracy = noisy_acc
            # Save using the specific filename set in wandb config
            torch.save(model.state_dict(), wandb.config.best_model_filename)
            print(f"New best model saved as '{wandb.config.best_model_filename}' with accuracy: {best_accuracy:.2f}%")
            wandb.run.summary["best_val_accuracy_noisy"] = best_accuracy

    print("Training completed.")

def test_gaussian(model, loader_clean, loader_noise1, loader_noise2, device, std1=0.5, std2=1.0):
    clean_accuracy = evaluate_model(model, loader_clean, device, "Baseline (Clean Images)")
    noise1_accuracy = 0
    noise2_accuracy = 0
    for _ in range(5):  # Run multiple evaluations to average out randomness
        noise1_accuracy += evaluate_model(model, loader_noise1, device, f"Gaussian Noise (std={std1})")
        noise2_accuracy += evaluate_model(model, loader_noise2, device, f"Gaussian Noise (std={std2})")
    print(f"Average Accuracy for Clean Images: {clean_accuracy:.2f}%")
    print(f"Average Accuracy for Gaussian Noise (std={std1}): {noise1_accuracy / 5:.2f}%")
    print(f"Average Accuracy for Gaussian Noise (std={std2}): {noise2_accuracy / 5:.2f}%")

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
                    for labels in IMAGENETTE_CLASSES.keys():
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
    train_loader, val_loader, test_loader = get_traing_val_test_loaders_for_gaussian(batch_size=32, std1=0.5, std2=1.0, data_name="gtsrb")