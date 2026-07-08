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

def main(dataset_name, model_name, group_norm, unet, gaussian, models_location = str(Path(__file__).parent)+"/models"):
    if gaussian:
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_gaussian(batch_size=32, std1=0.5, std2=1.0, data_name=dataset_name)
    else:
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_motion_blur(batch_size=32, kernel_size1=101, kernel_size2=151, data_name=dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name,group_norm,unet,models_location)
    model = model.to(device)
    model.eval()
    if(unet):
        model_visualizer = ResNet18FeatureVisualizer(model.get_base_model(), unet=model.get_unet())
    else:
        model_visualizer = ResNet18FeatureVisualizer(model)
    if gaussian:
        saving_location = str(Path(__file__).parent)+"/analysis_results/gaussian/"+model_name
    else:
        saving_location = str(Path(__file__).parent)+"/analysis_results/motion/"+model_name
    #test_gaussian(model, loader_clean, loader_noise1, loader_noise2, device, std1=0.5, std2=1.0)
    save_figures(dataset_name, model, model_visualizer, loader_clean, loader_noise1, loader_noise2, device, saving_location, max_samples=5)
    # save_features(model,model_visualizer, loader_clean, loader_noise1, loader_noise2, device, saving_location)
if __name__ == "__main__":
    model = load_model("gtsrb_gaussian_resnet18_prob0.5_group_norm0_Unet_False.pth", group_norm=0, unet=False, models_location = str(Path(__file__).parent.parent))
    loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_motion_blur(batch_size=32, kernel_size1=101, kernel_size2=151, data_name="gtsrb")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    evaluate_model(model, loader_noise1, device, "Baseline (Clean Images)")
    evaluate_model(model, loader_noise2, device, f"Motion Blur Noise (kernel_size=101)")
    main("gtsrb","gtsrb_gaussian_resnet18_prob0_group_norm0_Unet_False.pth",group_norm = 0, unet=False,gaussian = True, models_location = str(Path(__file__).parent.parent))
    #save_fft_map_for_an_index("imagenette","base.pth",group_norm = 0, unet=False, index=50,gaussian = True, load_model=load_model, saving_location = str(Path(__file__).parent)+"/analysis_results",models_location = str(Path(__file__).parent)+"/models")