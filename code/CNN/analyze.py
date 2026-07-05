from pathlib import Path
import sys
import timm
from visualizer import *
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from archtechre_common import *
from Unet import UNetWrapper
from network import CNN
from resnet_18.visualizer import replace_bn_with_gn



def main(model_name,group_norm,unet,gaussian,models_location = str(Path(__file__).parent)+"/models"): 
    if gaussian:
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_gaussian(batch_size=32, std1=0.5, std2=1.0)
    else:
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_motion_blur(batch_size=32, kernel_size1=20, kernel_size2=30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    if group_norm > 0:
        print(f"Replacing BatchNorm with GroupNorm (groups={group_norm})...")
        model = replace_bn_with_gn(model, num_groups=group_norm)
    if unet:
        print("Wrapping the model with UNet...")
        model = UNetWrapper(base_model=model)
    model.load_state_dict(torch.load(models_location+f"/{model_name}"))
    model = model.to(device)
    model.eval()
    if(unet):
        model_visualizer = CNNFeatureVisualizer(model.get_base_model(), unet=model.get_unet())
    else:
        model_visualizer = CNNFeatureVisualizer(model)
    if gaussian:
        saving_location = str(Path(__file__).parent)+"/analysis_results/gaussian"+model_name
    else:
        saving_location = str(Path(__file__).parent)+"/analysis_results/motion"+model_name
    #test_gaussian(model, loader_clean, loader_noise1, loader_noise2, device, std1=0.5, std2=1.0)
    save_figures(model, model_visualizer, loader_clean, loader_noise1, loader_noise2, device, saving_location, max_samples=5)
    # save_features(model,model_visualizer, loader_clean, loader_noise1, loader_noise2, device, saving_location)
if __name__ == "__main__":
    main("motion_cnn_prob0.5_group_norm0_Unet_False.pth",group_norm = 0, unet=False,gaussian = False)