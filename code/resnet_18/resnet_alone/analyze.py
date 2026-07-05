from pathlib import Path
import sys

import matplotlib
import torchvision
import tqdm
    
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from resnet_18.train_motion import *
from visulaizer import *

def test():
    clean_accuracy = evaluate_model(model, loader_clean, device, "Baseline (Clean Images)")
    noise1_accuracy = 0
    noise2_accuracy = 0
    for _ in range(5):  # Run multiple evaluations to average out randomness
        noise1_accuracy += evaluate_model(model, loader_noise1, device, f"Gaussian Noise (std={std1})")
        noise2_accuracy += evaluate_model(model, loader_noise2, device, f"Gaussian Noise (std={std2})")
    print(f"Average Accuracy for Clean Images: {clean_accuracy:.2f}%")
    print(f"Average Accuracy for Gaussian Noise (std={std1}): {noise1_accuracy / 5:.2f}%")
    print(f"Average Accuracy for Gaussian Noise (std={std2}): {noise2_accuracy / 5:.2f}%")

def analyze(index = 0):
    # Load a few images from the clean dataset for visualization
    
    sample_images = []
    sample_tensors = []
    sample_images.append(denormalize(dataset_clean[index][0]).permute(1, 2, 0).numpy())  # Convert tensor to HWC format for visualization
    sample_tensors.append(dataset_clean[index][0].unsqueeze(0).to(device))  # Add batch dimension and move to device
    sample_images.append(denormalize(dataset_noise1[index][0]).permute(1, 2, 0).numpy())
    sample_tensors.append(dataset_noise1[index][0].unsqueeze(0).to(device))
    sample_images.append(denormalize(dataset_noise2[index][0]).permute(1, 2, 0).numpy())
    sample_tensors.append(dataset_noise2[index][0].unsqueeze(0).to(device))
    labels = []
    # Get predictions for each image
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for tensor in sample_tensors:
            output = model(tensor)
            pred_class = torch.argmax(output, dim=1).item()
            labels.append(get_class_name(pred_class))  # Convert class index to human-readable label
    
    fig = display_multiple_images_progress(model, sample_tensors, sample_images, labels)
    return fig  # Return the figure for further processing or saving if needed


if __name__ == "__main__":
    # Set device
    BASE_DIR = Path(__file__).resolve().parent
    saving_location = str(BASE_DIR)  +"\\analysis_results"
    saving_location = Path(saving_location)  # Ensure it's a Path object
    saving_location.mkdir(exist_ok=True)
    model_name = "base_groupnorm16_porb5_noise5"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    std1 = 0.5
    std2 = 1.0
    # Load the model
    train_dir, val_dir = download_and_extract_imagenette(data_dir="./data")
    model = torchvision.models.resnet18()
    model = replace_bn_with_gn(model, num_groups=16)  # Replace BatchNorm with GroupNorm
    # 1. Get the absolute path to the directory where this python script lives
    

    # 2. Build the path from the script's location
    model_path = BASE_DIR / "models" / f"{model_name}.pth"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    figer = plot_layer_kernels(model.conv1, begin_idx=0, end_idx=128)  # Visualize the first 64 kernels of the first convolutional layer
    (saving_location/model_name).mkdir(parents=True, exist_ok=True)
    figer.savefig(str(saving_location/model_name) + f"/conv1_kernels.png", bbox_inches='tight')
    # Load the dataset

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
    loader_clean = DataLoader(dataset_clean, batch_size=32, shuffle=False, num_workers=2)

    dataset_noise1 = ImageFolder(root=val_dir, transform=transform_noise_std1, target_transform=map_class_to_imagenet)
    loader_noise1 = DataLoader(dataset_noise1, batch_size=32, shuffle=False, num_workers=2)

    dataset_noise2 = ImageFolder(root=val_dir, transform=transform_noise_std2, target_transform=map_class_to_imagenet)
    loader_noise2 = DataLoader(dataset_noise2, batch_size=32, shuffle=False, num_workers=2)

    #analyze()
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

    print("dataset clean length: ", len(dataset_clean))
    model.eval()  # Set model to evaluation mode
    for i in tqdm.tqdm(range(len(dataset_clean))):  # Analyze the first three images
        output_clean = model(dataset_clean[i][0].unsqueeze(0).to(device))
        output_noise1 = model(dataset_noise1[i][0].unsqueeze(0).to(device))
        output_noise2 = model(dataset_noise2[i][0].unsqueeze(0).to(device))
        predction1 = torch.argmax(output_clean, dim=1).item()
        predction2 = torch.argmax(output_noise1, dim=1).item()
        predction3 = torch.argmax(output_noise2, dim=1).item()
        label1 = dataset_clean[i][1]
        label2 = dataset_noise1[i][1]
        label3 = dataset_noise2[i][1]
        flag1 = predction1 == label1
        flag2 = predction2 == label2
        flag3 = predction3 == label3


        
        if dictionary[flag1][flag2][flag3][dataset_clean[i][1]] < 10:
            print(f"Analyzing image index {i} with label {dataset_clean[i][1]}...")
            fig  = analyze(i)
            location = saving_location/model_name/f"clean_{flag1}_noise1_{flag2}_noise2_{flag3}/real_label_{dataset_clean[i][1]}"
            location.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(saving_location/model_name) + f"/clean_{flag1}_noise1_{flag2}_noise2_{flag3}/real_label_{dataset_clean[i][1]}/image_{dictionary[flag1][flag2][flag3][dataset_clean[i][1]]}.png", bbox_inches='tight')
            matplotlib.pyplot.close()
            dictionary[flag1][flag2][flag3][dataset_clean[i][1]] += 1