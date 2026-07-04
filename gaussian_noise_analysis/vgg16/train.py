import wandb
import torch
import torch.nn as nn
from torchvision import models
import sys
from pathlib import Path

import network

from visualizer import*

parent_dir = str(Path(__file__).parent.parent)

# 2. Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from archtechre_common import *
from Unet import UNetWrapper
from resnet_18.visualizer import replace_bn_with_gn

def main(prob, group_norm,Unet):
    wandb.init(
    project="CNN",
    name="gaussian_CNN_prob{}_group_norm{}_Unet_{}".format(prob, group_norm, Unet),
    config={
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "batch_size": 32,
        "num_workers": 2,
        "seed": 42,
        "train_split_ratio": 0.8,
        "image_resize": 256,
        "image_crop": 224,
        "train_noise_std": 0.5,
        "train_noise_prob": prob,
        "eval_noise_std1": 0.5,
        "eval_noise_std2": 1.0,
        "best_model_filename": "gaussian_CNN_prob{}_group_norm{}_Unet_{}.pth".format(prob, group_norm, Unet),
        "plot_every_n_epochs": 1,
        "group_norm_groups": group_norm,
        "UNet": Unet
    }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2 =get_traing_val_test_loaders_for_gaussian(config=config)
    print("Downloading/Loading pretrained VGG16...")
    #weights = models.VGG16_BN_Weights.DEFAULT
    #model = models.vgg16_bn(weights=weights)
    model = network.CNN(begin_features=128, group_channels=64, num_classes=1000)
    
    if config.group_norm_groups > 0:
        print(f"Replacing BatchNorm with GroupNorm (groups={config.group_norm_groups})...")
        model = replace_bn_with_gn(model, num_groups=config.group_norm_groups)
    if config.UNet:
        print("Wrapping the model with UNet...")
        model = UNetWrapper(base_model=model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    if config.UNet:
        model_visualizer = VGG16FeatureVisualizer(model.get_base_model(), unet=model.get_unet())
    else:
        model_visualizer = VGG16FeatureVisualizer(model)

    # 6. Train and finish
    train_model(model, train_loader, val_loader, val_loader2, val_loader3, criterion, optimizer, device,prog_vis =model_visualizer, config=config)

    model.load_state_dict(torch.load(config.best_model_filename))
    test_acc_clean = evaluate_model(model, loader_clean, device, description="Final Test on Clean Dataset")
    test_acc_noisy1 = evaluate_model(model, loader_noise1, device, description=f"Final Test on Noisy Dataset (std={config.eval_noise_std1})")
    test_acc_noisy2 = evaluate_model(model, loader_noise2, device, description=f"Final Test on Noisy Dataset (std={config.eval_noise_std2})")
    wandb.run.summary["final_test_accuracy_clean"] = test_acc_clean
    wandb.run.summary["final_test_accuracy_noisy1 std={config.eval_noise_std1}"] = test_acc_noisy1
    wandb.run.summary["final_test_accuracy_noisy2 std={config.eval_noise_std2}"] = test_acc_noisy2
    # End the wandb run
    print("Training completed. Ending wandb run.")
    wandb.finish()
    

if __name__ == "__main__":
    probs = [0,0.5]
    group_norms = [8,0, 16, 32]
    Unet_options = [True, False]
    for prob in probs:
        for group_norm in group_norms:
            for Unet in Unet_options:
                try:
                    main(prob=prob, group_norm=group_norm, Unet=Unet)
                except Exception as e:
                    print(f"An error occurred during training with prob={prob}, group_norm={group_norm}, Unet={Unet}: {e}")
                    continue