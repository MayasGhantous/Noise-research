from pathlib import Path
import sys
from visualizer import *
from vit.visualizer import replace_vit_layernorm_with_groupnorm
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from torchvision import models
from visualizer_common import *
from Unet import UNetWrapper


def load_model(model_name,group_norm,unet,models_location):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if group_norm > 0:
        print(f"Replacing BatchNorm with GroupNorm (groups={group_norm})...")
        model = replace_vit_layernorm_with_groupnorm(model, num_groups=group_norm)
    if unet:
        print("Wrapping the model with UNet...")
        model = UNetWrapper(base_model=model)
    model.load_state_dict(torch.load(models_location+f"/{model_name}"))
    return model

def main(dataset_name, model_name, group_norm, unet, noise_type, models_location = str(Path(__file__).parent)+"/models"):
    if noise_type == "gaussian":
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_gaussian(batch_size=32, std1=0.5, std2=1.0, data_name=dataset_name)
    elif noise_type == "motion_blur":
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_motion_blur(batch_size=32, kernel_size1=51, kernel_size2=91, data_name=dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name,group_norm,unet,models_location)
    model = model.to(device)
    model.eval()
    if(unet):
        model_visualizer = ResNet18FeatureVisualizer(model.get_base_model(), unet=model.get_unet())
    else:
        model_visualizer = ResNet18FeatureVisualizer(model)
    if noise_type == "gaussian":
        saving_location = str(Path(__file__).parent)+"/analysis_results/gaussian/"+model_name
    elif noise_type == "motion_blur":
        saving_location = str(Path(__file__).parent)+"/analysis_results/motion/"+model_name
    #test_gaussian(model, loader_clean, loader_noise1, loader_noise2, device, std1=0.5, std2=1.0)
    save_figures(dataset_name, model, model_visualizer, loader_clean, loader_noise1, loader_noise2, device, saving_location, max_samples=5)
    # save_features(model,model_visualizer, loader_clean, loader_noise1, loader_noise2, device, saving_location)
if __name__ == "__main__":
    model_name = "imagenette_motion_blur_resnet18_group_norm0_Unet_False_pretrained.pth"
    dataset_name = "imagenette"
    unet = False
    group_norm = 0
    noise_type = "motion_blur"
    model = load_model(model_name, group_norm=group_norm, unet=unet, models_location = str(Path(__file__).parent)+"/models")
    if noise_type == "gaussian":
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_gaussian(batch_size=32, std1=0.5, std2=1.0, data_name=dataset_name)
    elif noise_type == "motion_blur":
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_motion_blur(batch_size=32, kernel_size1=51, kernel_size2=91, data_name=dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    evaluate_model(model, loader_clean, device, "Baseline (Clean Images)")
    evaluate_model(model, loader_noise1, device, "Motion Blur Noise (kernel_size=51)")
    evaluate_model(model, loader_noise2, device, f"Motion Blur Noise (kernel_size=91)")

    main(dataset_name, model_name, group_norm = 0, unet=False,noise_type = noise_type, models_location = str(Path(__file__).parent)+"/models")
    #save_fft_map_for_an_index("gtsrb","gtsrb_motion_blur_resnet18_base_line.pth",group_norm = 0, unet=False, index=9988,gaussian = False, load_model=load_model, saving_location = str(Path(__file__).parent)+"/analysis_results",models_location = str(Path(__file__).parent))