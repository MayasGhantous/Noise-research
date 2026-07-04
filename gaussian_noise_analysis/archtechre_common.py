from pathlib import Path
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
import torchvision.transforms.functional as TF

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
    Custom transform to add motion blur to a tensor with a dynamic random angle.
    """
    def __init__(self, kernel_size=15, angle_range=(0.0, 360.0)):
        """
        Args:
            kernel_size (int): The intensity/length of the blur. Must be an odd number.
            angle_range (tuple or float): A tuple (min_angle, max_angle) for random angles, 
                                          or a single float for a fixed angle.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.kernel_size = kernel_size
        self.angle_range = angle_range
        
    def __call__(self, tensor):
        # 1. Determine the angle dynamically for THIS specific image
        if isinstance(self.angle_range, (tuple, list)):
            current_angle = random.uniform(self.angle_range[0], self.angle_range[1])
        else:
            current_angle = float(self.angle_range)

        channels, height, width = tensor.shape
        
        # 2. Create and rotate the kernel
        kernel = torch.zeros((1, self.kernel_size, self.kernel_size), dtype=torch.float32, device=tensor.device)
        kernel[0, self.kernel_size // 2, :] = 1.0
        kernel = TF.rotate(kernel, current_angle, interpolation=TF.InterpolationMode.BILINEAR)
        kernel = kernel / torch.sum(kernel)
        
        # 3. Apply the convolution
        kernel = kernel.expand(channels, 1, self.kernel_size, self.kernel_size)
        tensor_batched = tensor.unsqueeze(0)
        padding = self.kernel_size // 2
        
        blurred_tensor = F.conv2d(tensor_batched, kernel, padding=padding, groups=channels)
        
        return blurred_tensor.squeeze(0) 
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


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


def get_test_loaders_for_gaussian(batch_size=32,std1=0.5,std2=1):

    _, val_dir = download_and_extract_imagenette()

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



def get_test_loaders_for_motion_blur(batch_size=32,kernel_size1=15,kernel_size2=25):

    _, val_dir = download_and_extract_imagenette()

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
    
    dataset_clean = ImageFolder(root=val_dir, transform=transform_clean, target_transform=map_class_to_imagenet)
    loader_clean = DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=2)

    dataset_noise1 = ImageFolder(root=val_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=batch_size, shuffle=False, num_workers=2)

    dataset_noise2 = ImageFolder(root=val_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
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
    train_dir, test_dir = download_and_extract_imagenette()

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

    train_dataset1 = ImageFolder(root=train_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
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


    train_dataset2 = ImageFolder(root=train_dir, transform=transform_clean, target_transform=map_class_to_imagenet)
    _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)#clean validation set
    _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)#noisey validation
    train_dataset3 = ImageFolder(root=train_dir, transform=train_transform, target_transform=map_class_to_imagenet)
    train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)#train

    train_dataset4 = ImageFolder(root=train_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
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
    
    dataset_clean = ImageFolder(root=test_dir, transform=transform_clean, target_transform=map_class_to_imagenet)

    loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    dataset_noise1 = ImageFolder(root=test_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    dataset_noise2 = ImageFolder(root=test_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2


def get_traing_val_test_loaders_for_motion_blure(config):
    train_dir, test_dir = download_and_extract_imagenette()

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

    train_dataset1 = ImageFolder(root=train_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
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


    train_dataset2 = ImageFolder(root=train_dir, transform=transform_clean, target_transform=map_class_to_imagenet)
    _, val_subset = train_val_split(train_dataset2, train_indices, val_indices)#clean validation set
    _, val2_subset = train_val_split(train_dataset1, train_indices, val_indices)#noisey validation
    train_dataset3 = ImageFolder(root=train_dir, transform=train_transform, target_transform=map_class_to_imagenet)
    train_subset, _ = train_val_split(train_dataset3, train_indices, val_indices)#train

    train_dataset4 = ImageFolder(root=train_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
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
    
    dataset_clean = ImageFolder(root=test_dir, transform=transform_clean, target_transform=map_class_to_imagenet)

    loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    dataset_noise1 = ImageFolder(root=test_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    dataset_noise2 = ImageFolder(root=test_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
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
            wandb.log({f"Network Progression (Feature Map 5)": wandb.Image(fig)})
            plt.close(fig)

        if best_accuracy <= noisy_acc:
            best_accuracy = noisy_acc
            # Save using the specific filename set in wandb config
            torch.save(model.state_dict(), wandb.config.best_model_filename)
            print(f"New best model saved as '{wandb.config.best_model_filename}' with accuracy: {best_accuracy:.2f}%")
            wandb.run.summary["best_val_accuracy_noisy"] = best_accuracy

    print("Training completed.")

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
    clean,noisy1,noisy2 = get_test_loaders_for_motion_blur(batch_size=32)
    images = next(iter(clean))[0]
    images = images.squeeze(0)  # Get the first image from the batch
    plot_tensor_fft_channels(images)
    