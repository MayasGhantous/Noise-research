from pathlib import Path
import sys
import timm
from visualizer import *
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from visualizer_common import *
from Unet import UNetWrapper


def load_model(model_name,group_norm,unet,models_location):
    model = timm.create_model('vit_tiny_patch16_224')
    if group_norm > 0:
        print(f"Replacing BatchNorm with GroupNorm (groups={group_norm})...")
        model = replace_vit_layernorm_with_groupnorm(model, num_groups=group_norm)
    if unet:
        print("Wrapping the model with UNet...")
        model = UNetWrapper(base_model=model)
    model.load_state_dict(torch.load(models_location+f"/{model_name}"))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model



def main(dataset_name, model_name, group_norm, unet, noise_type, models_location = str(Path(__file__).parent)):
    if noise_type == "gaussian":
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_gaussian(batch_size=32, std1=0.5, std2=1.0, data_name=dataset_name)
    elif noise_type == "motion_blur":
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_motion_blur(batch_size=32, kernel_size1=51, kernel_size2=91, data_name=dataset_name)
    elif noise_type == "defocus_blur":
        loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_defocus(batch_size=32, rad1=10, rad2=25, data_name=dataset_name)
    model = load_model(model_name,group_norm,unet,models_location)
    model.eval()
    if(unet):
        model_visualizer = ViTBatchAttentionVisualizer(model.get_base_model(), unet=model.get_unet())
    else:
        model_visualizer = ViTBatchAttentionVisualizer(model)
    saving_location = str(Path(__file__).parent)+f"/analysis_results/{noise_type}/"+model_name
    #test_gaussian(model, loader_clean, loader_noise1, loader_noise2, device, std1=0.5, std2=1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_figures(model, model_visualizer, loader_clean, loader_noise1, loader_noise2,device , saving_location, max_samples=5)
    # save_features(model,model_visualizer, loader_clean, loader_noise1, loader_noise2, device, saving_location)


if __name__ == "__main__":
    data_sets = []
    models_name =[]
    group_norms = []
    unets = []
    noise_types = []
    for data_name,model_name, group_norm, unet, noise_type in zip(data_sets, models_name, group_norms, unets, noise_types):
        main(data_name, model_name, group_norm, unet, noise_type)
