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

def main(prob, group_norm,Unet,data_name,noise_type):
    wandb.init(
    project="Noise-Research",
    name="{}_CNN_group_norm{}_Unet_{}".format(noise_type, group_norm, Unet),
    config={
        "learning_rate": 1e-3,
        "num_epochs": 20,
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
        "kernel_size1": 20,
        "kernel_size2": 30,
        "best_model_filename": "{}_CNN_prob{}_group_norm{}_Unet_{}.pth".format(noise_type, prob, group_norm, Unet),
        "plot_every_n_epochs": 1,
        "group_norm_groups": group_norm,
        "UNet": Unet,
        "data_name": data_name,
        "noise_type": noise_type
    }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if config.noise_type == "gaussian":
        train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2 =get_traing_val_test_loaders_for_gaussian(config=config)
    elif config.noise_type == "motion_blur":
        train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2 =get_traing_val_test_loaders_for_motion_blure(config=config)
    #weights = models.VGG16_BN_Weights.DEFAULT
    #model = models.vgg16_bn(weights=weights)
    model = network.CNN( num_classes=1000)
    
    if config.group_norm_groups > 0:
        print(f"Replacing BatchNorm with GroupNorm (groups={config.group_norm_groups})...")
        model = replace_bn_with_gn(model, num_groups=config.group_norm_groups)
    if config.UNet:
        print("Wrapping the model with UNet...")
        model = UNetWrapper(base_model=model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    if config.UNet:
        model_visualizer = CNNFeatureVisualizer(model.get_base_model(), unet=model.get_unet())
    else:
        model_visualizer = CNNFeatureVisualizer(model)

    # 6. Train and finish
    train_model(model, train_loader, val_loader, val_loader2, val_loader3, criterion, optimizer, device,prog_vis =model_visualizer, config=config)

    model.load_state_dict(torch.load(config.best_model_filename))
    test_acc_clean = evaluate_model(model, loader_clean, device, description="Final Test on Clean Dataset")
    test_acc_noisy1 = evaluate_model(model, loader_noise1, device, description=f"Final Test on Noisy Dataset")
    test_acc_noisy2 = evaluate_model(model, loader_noise2, device, description=f"Final Test on higher Noise Dataset")
    wandb.run.summary["final_test_accuracy_clean"] = test_acc_clean
    wandb.run.summary["final_test_accuracy_noisy1"] = test_acc_noisy1
    wandb.run.summary["final_test_accuracy_noisy2"] = test_acc_noisy2
    # End the wandb run
    print("Training completed. Ending wandb run.")
    wandb.finish()
    

if __name__ == "__main__":
    main(prob=0, group_norm=0, Unet=False)
    probs = [0.5]
    group_norms = [0,8, 16]
    Unet_options = [False, True]
    for prob in probs:
        for group_norm in group_norms:
            for Unet in Unet_options:
                main(prob=prob, group_norm=group_norm, Unet=Unet)
