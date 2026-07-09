import os
from pathlib import Path
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm 
import wandb

from torchvision import transforms
from torchvision.datasets import Imagenette as TVImagenette, GTSRB as TVGTSRB
from torch.utils.data import DataLoader, Subset

IMAGENETTE = "imagenette"
GTSRB = "gtsrb"

# Create shared data directory path
PARENT_DIR = str(Path(__file__).parent.parent)
DATA_DIR = os.path.join(PARENT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

if not os.environ.get("SCRIPT_ALREADY_RAN"):
    print("setting seeds")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.environ["SCRIPT_ALREADY_RAN"] = "True"

# --- Class Mapping Dictionaries & Functions ---
IMAGENETTE_CLASSES = {
    0: 'Tench', 217: 'English Springer', 482: 'Cassette Player', 
    491: 'Chain Saw', 497: 'Church', 566: 'French Horn', 
    569: 'Garbage Truck', 571: 'Gas Pump', 574: 'Golf Ball', 701: 'Parachute'
}
GTSRB_CLASSES = {
    0: 'Speed limit (20kmh)', 1: 'Speed limit (30kmh)', 2: 'Speed limit (50kmh)', 
    3: 'Speed limit (60kmh)', 4: 'Speed limit (70kmh)', 5: 'Speed limit (80kmh)', 
    6: 'End of speed limit (80kmh)', 7: 'Speed limit (100kmh)', 8: 'Speed limit (120kmh)', 
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection', 
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 
    16: 'Veh over 3.5 tons prohibited', 17: 'No entry', 18: 'General caution', 
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve', 
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice_snow', 
    31: 'Wild animals crossing', 32: 'End speed + passing limits', 33: 'Turn right ahead', 
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right', 
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left', 
    40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh over 3.5 tons'
}
IMAGENETTE_TO_IMAGENET = {0:0, 1:217, 2:482, 3:491, 4:497, 5:566, 6:569, 7:571, 8:574, 9:701}

def map_class_to_imagenet(y):
    return IMAGENETTE_TO_IMAGENET[y]

def get_class_name(data_name, class_idx):
    if data_name == IMAGENETTE:
        return IMAGENETTE_CLASSES.get(class_idx, f"Class {class_idx}")
    elif data_name == GTSRB:
        return GTSRB_CLASSES.get(class_idx, f"Class {class_idx}")
    else:
        return f"Class {class_idx}"

# --- Custom Noise Transform ---
class AddMotionBlur(object):
    """
    Ultra-fast custom transform to add motion blur using OpenCV precomputed kernels.
    """
    def __init__(self, kernel_size=15, angle_range=(0.0, 360.0), num_angles=360):
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
            kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
            kernel[self.kernel_size // 2, :] = 1.0
            
            center = (self.kernel_size / 2 - 0.5, self.kernel_size / 2 - 0.5)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation_matrix, (self.kernel_size, self.kernel_size))
            
            kernel = kernel / np.sum(kernel)
            self.kernels.append(kernel)

    def __call__(self, tensor):
        idx = random.randint(0, len(self.kernels) - 1)
        kernel = self.kernels[idx]
        
        img_np = tensor.permute(1, 2, 0).numpy()
        blurred_np = cv2.filter2D(img_np, -1, kernel)
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
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

# Algorithm taken from ImageNetC code
class AddDefocusBlur(object):
    def __init__(self, radius):
        self.c = (radius,0.5)

    def __call__(self, tensor):
        tensor = np.array(tensor)
        kernel = self.disk(radius=self.c[0], alias_blur=self.c[1])
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(tensor[d, :, :], -1, kernel))
        channels = np.array(channels)  
        return torch.from_numpy(np.clip(channels, 0, 1))
    
    def disk(self, radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
    
# --- Evaluation Function ---
def evaluate_model(model, dataloader, device, description=""):
    correct = 0
    total = 0
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

def get_test_loaders_for_gaussian(batch_size=32, std1=0.5, std2=1, data_name=IMAGENETTE):
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_clean = transforms.Compose([*base_transforms, normalization])
    transform_noise_std1 = transforms.Compose([*base_transforms, AddGaussianNoise(mean=0.0, std=std1), normalization])
    transform_noise_std2 = transforms.Compose([*base_transforms, AddGaussianNoise(mean=0.0, std=std2), normalization])

    print("Loading validation datasets with different noise profiles...")
    if data_name == IMAGENETTE:
        dataset_clean = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_clean, target_transform=map_class_to_imagenet)
        dataset_noise1 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
        dataset_noise2 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
    elif data_name == GTSRB:
        dataset_clean = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_clean)
        dataset_noise1 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_noise_std1)
        dataset_noise2 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_noise_std2)

    loader_clean = DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=2)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=batch_size, shuffle=False, num_workers=2)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=batch_size, shuffle=False, num_workers=2)

    return loader_clean, loader_noise1, loader_noise2


def get_test_loaders_for_motion_blur(batch_size=32, kernel_size1=15, kernel_size2=25, data_name=IMAGENETTE):
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_clean = transforms.Compose([*base_transforms, normalization])
    transform_noise_std1 = transforms.Compose([*base_transforms, AddMotionBlur(kernel_size=kernel_size1), normalization])
    transform_noise_std2 = transforms.Compose([*base_transforms, AddMotionBlur(kernel_size=kernel_size2), normalization])

    print("Loading validation datasets with different noise profiles...")
    if data_name == IMAGENETTE:
        dataset_clean = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_clean, target_transform=map_class_to_imagenet)
        dataset_noise1 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
        dataset_noise2 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
    else:
        dataset_clean = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_clean)
        dataset_noise1 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_noise_std1)
        dataset_noise2 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_noise_std2)

    loader_clean = DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=2)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=batch_size, shuffle=False, num_workers=2)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=batch_size, shuffle=False, num_workers=2)

    return loader_clean, loader_noise1, loader_noise2

def get_test_loaders_for_defocus(batch_size=32, rad1=10, rad2=25, data_name=IMAGENETTE):
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_clean = transforms.Compose([*base_transforms, normalization])
    transform_defocus_rad1 = transforms.Compose([*base_transforms, AddDefocusBlur(radius=rad1), normalization])
    transform_defocus_rad2 = transforms.Compose([*base_transforms, AddDefocusBlur(radius=rad2), normalization])

    print("Loading validation datasets with different noise profiles...")
    if data_name == IMAGENETTE:
        dataset_clean = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_clean, target_transform=map_class_to_imagenet)
        dataset_noise1 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_defocus_rad1, target_transform=map_class_to_imagenet)
        dataset_noise2 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_defocus_rad2, target_transform=map_class_to_imagenet)
    elif data_name == GTSRB:
        dataset_clean = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_clean)
        dataset_noise1 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_defocus_rad1)
        dataset_noise2 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_defocus_rad2)

    loader_clean = DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=2)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=batch_size, shuffle=False, num_workers=2)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=batch_size, shuffle=False, num_workers=2)

    return loader_clean, loader_noise1, loader_noise2

def train_val_split(dataset, train_indices, val_indices):
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    return train_subset, val_subset

def get_traing_val_test_loaders_for_gaussian(config):
    base_transforms = [
        transforms.Resize(config.image_resize),
        transforms.CenterCrop(config.image_crop),
        transforms.ToTensor(),
    ]
    
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_clean = transforms.Compose([*base_transforms, normalization])
    transform_noise_std1 = transforms.Compose([*base_transforms, AddGaussianNoise(mean=0.0, std=config.eval_noise_std1), normalization])
    transform_noise_std2 = transforms.Compose([*base_transforms, AddGaussianNoise(mean=0.0, std=config.eval_noise_std2), normalization])
    
    train_transform = transforms.Compose([
        *base_transforms,
        transforms.RandomApply([AddGaussianNoise(std=config.train_noise_std)], p=config.train_noise_prob), 
        normalization
    ])

    if config.data_name == IMAGENETTE:
        train_dataset1 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=True, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    else:
        train_dataset1 = TVGTSRB(root=DATA_DIR, split='train', download=True, transform=transform_noise_std1)
                                     
    dataset_size = len(train_dataset1)
    train_size = int(config.train_split_ratio * dataset_size)    
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    if config.data_name == IMAGENETTE:
        train_dataset2 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=False, transform=transform_clean, target_transform=map_class_to_imagenet)
        _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)
        _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)
        
        train_dataset3 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=False, transform=train_transform, target_transform=map_class_to_imagenet)
        train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)

        train_dataset4 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=False, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
        _, val3_subset = train_val_split(train_dataset4, train_indices, val_indices)
    else:
        train_dataset2 = TVGTSRB(root=DATA_DIR, split='train', download=False, transform=transform_clean)
        _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)
        _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)
        
        train_dataset3 = TVGTSRB(root=DATA_DIR, split='train', download=False, transform=train_transform)
        train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)

        train_dataset4 = TVGTSRB(root=DATA_DIR, split='train', download=False, transform=transform_noise_std2)
        _, val3_subset = train_val_split(train_dataset4, train_indices, val_indices)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    val_loader2 = DataLoader(val2_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    val_loader3 = DataLoader(val3_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    print("Loading validation datasets with different noise profiles...")
    if config.data_name == IMAGENETTE:
        dataset_clean = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_clean, target_transform=map_class_to_imagenet)
        dataset_noise1 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
        dataset_noise2 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
    else:
        dataset_clean = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_clean)
        dataset_noise1 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_noise_std1)
        dataset_noise2 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_noise_std2)
        
    loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    return train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2

def get_traing_val_test_loaders_for_motion_blure(config):
    base_transforms = [
        transforms.Resize(config.image_resize),
        transforms.CenterCrop(config.image_crop),
        transforms.ToTensor(),
    ]
    
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_clean = transforms.Compose([*base_transforms, normalization])
    transform_noise_std1 = transforms.Compose([*base_transforms, AddMotionBlur(kernel_size=config.kernel_size1), normalization])
    transform_noise_std2 = transforms.Compose([*base_transforms, AddMotionBlur(kernel_size=config.kernel_size2), normalization])
    
    train_transform = transforms.Compose([
        *base_transforms,
        transforms.RandomApply([AddMotionBlur(kernel_size=config.kernel_size1)], p=config.train_noise_prob), 
        normalization
    ])

    if config.data_name == IMAGENETTE:
        train_dataset1 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=True, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    else:
        train_dataset1 = TVGTSRB(root=DATA_DIR, split='train', download=True, transform=transform_noise_std1)
        
    dataset_size = len(train_dataset1)
    train_size = int(config.train_split_ratio * dataset_size)    
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    if config.data_name == IMAGENETTE:
        train_dataset2 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=False, transform=transform_clean, target_transform=map_class_to_imagenet)
        _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)
        _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)
        
        train_dataset3 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=False, transform=train_transform, target_transform=map_class_to_imagenet)
        train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)

        train_dataset4 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=False, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
        _, val3_subset = train_val_split(train_dataset4, train_indices, val_indices)
    else:
        train_dataset2 = TVGTSRB(root=DATA_DIR, split='train', download=False, transform=transform_clean)
        _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)
        _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)
        
        train_dataset3 = TVGTSRB(root=DATA_DIR, split='train', download=False, transform=train_transform)
        train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)

        train_dataset4 = TVGTSRB(root=DATA_DIR, split='train', download=False, transform=transform_noise_std2)
        _, val3_subset = train_val_split(train_dataset4, train_indices, val_indices)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    val_loader2 = DataLoader(val2_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    val_loader3 = DataLoader(val3_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    print("Loading validation datasets with different noise profiles...")
    if config.data_name == IMAGENETTE:
        dataset_clean = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_clean, target_transform=map_class_to_imagenet)
        dataset_noise1 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
        dataset_noise2 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
    else:
        dataset_clean = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_clean)
        dataset_noise1 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_noise_std1)
        dataset_noise2 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_noise_std2)
        
    loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    return train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2

def get_traing_val_test_loaders_for_defocus_blur(config):
    base_transforms = [
        transforms.Resize(config.image_resize),
        transforms.CenterCrop(config.image_crop),
        transforms.ToTensor(),
    ]
    
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_clean = transforms.Compose([*base_transforms, normalization])
    transform_defocus_rad1 = transforms.Compose([*base_transforms, AddDefocusBlur(radius=config.radius1), normalization])
    transform_defocus_rad2 = transforms.Compose([*base_transforms, AddDefocusBlur(radius=config.radius2), normalization])
    
    train_transform = transforms.Compose([
        *base_transforms,
        transforms.RandomApply([AddDefocusBlur(radius=config.radius1)], p=config.train_noise_prob), 
        normalization
    ])

    if config.data_name == IMAGENETTE:
        train_dataset1 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=True, transform=transform_defocus_rad1, target_transform=map_class_to_imagenet)
    else:
        train_dataset1 = TVGTSRB(root=DATA_DIR, split='train', download=True, transform=transform_defocus_rad1)
        
    dataset_size = len(train_dataset1)
    train_size = int(config.train_split_ratio * dataset_size)    
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    if config.data_name == IMAGENETTE:
        train_dataset2 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=False, transform=transform_clean, target_transform=map_class_to_imagenet)
        _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)
        _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)
        
        train_dataset3 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=False, transform=train_transform, target_transform=map_class_to_imagenet)
        train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)

        train_dataset4 = TVImagenette(root=DATA_DIR, split='train', size='160px', download=False, transform=transform_defocus_rad2, target_transform=map_class_to_imagenet)
        _, val3_subset = train_val_split(train_dataset4, train_indices, val_indices)
    else:
        train_dataset2 = TVGTSRB(root=DATA_DIR, split='train', download=False, transform=transform_clean)
        _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)
        _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)
        
        train_dataset3 = TVGTSRB(root=DATA_DIR, split='train', download=False, transform=train_transform)
        train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)

        train_dataset4 = TVGTSRB(root=DATA_DIR, split='train', download=False, transform=transform_defocus_rad2)
        _, val3_subset = train_val_split(train_dataset4, train_indices, val_indices)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    val_loader2 = DataLoader(val2_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    val_loader3 = DataLoader(val3_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    print("Loading validation datasets with different noise profiles...")
    if config.data_name == IMAGENETTE:
        dataset_clean = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_clean, target_transform=map_class_to_imagenet)
        dataset_noise1 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_defocus_rad1, target_transform=map_class_to_imagenet)
        dataset_noise2 = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True, transform=transform_defocus_rad2, target_transform=map_class_to_imagenet)
    else:
        dataset_clean = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_clean)
        dataset_noise1 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_defocus_rad1)
        dataset_noise2 = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_defocus_rad2)
        
    loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    return train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2

def train_model(model, train_loader, val_loader, val_loader2, val_loader3, criterion, optimizer, device, prog_vis=None, config=None):
    print("\nStarting training...")
    best_accuracy = evaluate_model(model, val_loader2, device, description="Initial Validation Accuracy")
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

        clean_acc = evaluate_model(model, val_loader, device, description=f"Validation after Epoch {epoch + 1}")
        noisy_acc = evaluate_model(model, val_loader2, device, description=f"Validation with Noise after Epoch {epoch + 1}")
        higher_order_acc = evaluate_model(model, val_loader3, device, description=f"Validation with Higher Order Noise after Epoch {epoch + 1}")
        
        wandb.log({
            "Epoch_accuracy": epoch_accuracy,
            "Clean Validation Accuracy": clean_acc,
            "Noisy Validation Accuracy": noisy_acc,
            "Higher Order Validation Accuracy": higher_order_acc,
            "Epoch Training Loss": epoch_loss / len(train_loader)
        })

        if prog_vis and (epoch + 1) % plot_every_n_epochs == 0:
            rand_idx = random.randint(0, len(val_loader.dataset) - 1)
            img, true_label = val_loader.dataset[rand_idx]
            img2, _ = val_loader2.dataset[rand_idx]
            img3, _ = val_loader3.dataset[rand_idx]
            
            img_clean = img.squeeze(0).to(device)
            img_noisy = img2.squeeze(0).to(device)
            img_higher_order = img3.squeeze(0).to(device)
            
            fig = prog_vis.extract_and_return_figure(config.data_name, torch.stack([img_clean, img_noisy, img_higher_order]), [true_label, true_label, true_label])
            
            wandb.log({f"Network Progression ": wandb.Image(fig)})
            plt.close(fig)

        if best_accuracy <= noisy_acc:
            best_accuracy = noisy_acc
            torch.save(model.state_dict(), wandb.config.best_model_filename)
            print(f"New best model saved as '{wandb.config.best_model_filename}' with accuracy: {best_accuracy:.2f}%")
            wandb.run.summary["best_val_accuracy_noisy"] = best_accuracy

    print("Training completed.")

def test_gaussian(model, loader_clean, loader_noise1, loader_noise2, device, std1=0.5, std2=1.0):
    clean_accuracy = evaluate_model(model, loader_clean, device, "Baseline (Clean Images)")
    noise1_accuracy = 0
    noise2_accuracy = 0
    for _ in range(2): 
        noise1_accuracy += evaluate_model(model, loader_noise1, device, f"Gaussian Noise (std={std1})")
        noise2_accuracy += evaluate_model(model, loader_noise2, device, f"Gaussian Noise (std={std2})")
    print(f"Average Accuracy for Clean Images: {clean_accuracy:.2f}%")
    print(f"Average Accuracy for Gaussian Noise (std={std1}): {noise1_accuracy / 2:.2f}%")
    print(f"Average Accuracy for Gaussian Noise (std={std2}): {noise2_accuracy / 2:.2f}%")
if __name__ == "__main__":
    data_loaders1, data_loaders2, data_loaders3 = get_test_loaders_for_gaussian(data_name=GTSRB)
    # 1. Grab a single batch of training data
    dataiter = iter(data_loaders1)
    images, labels = next(dataiter)

    # 2. Select the very first image and label in the batch
    first_image = images[0]
    first_label = labels[0]

    # 3. Denormalize the image using the function already in your script
    # (Otherwise, the colors will look completely washed out and weird)
    clean_image = denormalize(first_image)

    # 4. Convert the PyTorch Tensor [C, H, W] to a Numpy Array [H, W, C] for Matplotlib
    np_img = clean_image.permute(1, 2, 0).cpu().numpy()

    # 5. Plot the image
    plt.figure(figsize=(4, 4))
    plt.imshow(np_img)
    plt.title(f"Label: {get_class_name(IMAGENETTE, first_label.item())}")
    plt.axis('off')
    plt.show()
