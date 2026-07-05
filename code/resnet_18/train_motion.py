from visualizer import *
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)

# 2. Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Unet import  UNetWrapper

def main(prob,group_norm,unet):
    wandb.init(
        project="Resnet-18",
        name="motion_resnet18_prob{}_group_norm{}_Unet_{}".format(prob, group_norm, unet),
        config={
            "learning_rate": 1e-4,
            "num_epochs": 40,
            "batch_size": 32,
            "num_workers": 2,
            "seed": 42,
            "train_split_ratio": 0.8,
            "image_resize": 256,
            "image_crop": 224,
            "train_noise_prob": prob,
            "kernel_size1": 20,
            "kernel_size2": 30,
            "best_model_filename": "motion_resnet18_prob{}_group_norm{}_Unet_{}.pth".format(prob, group_norm, unet),
            "plot_every_n_epochs": 1,
            "group_norm_groups": group_norm,
            "UNet": unet
        }
    )
    
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2 =get_traing_val_test_loaders_for_motion_blure(config=config)
    print("Downloading/Loading pretrained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if config.group_norm_groups > 0:
        print(f"Replacing BatchNorm with GroupNorm (groups={config.group_norm_groups})...")
        model = replace_bn_with_gn(model, num_groups=config.group_norm_groups)
    if config.UNet:
        print("Wrapping the model with UNet...")
        model = UNetWrapper(base_model=model)
    
    model = model.to(device)
    
    if config.UNet:
        model_visualizer = ResNet18FeatureVisualizer(model.get_base_model(), unet=model.get_unet())
    else:
        model_visualizer = ResNet18FeatureVisualizer(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, val_loader, val_loader2, val_loader3, criterion, optimizer, device,prog_vis =model_visualizer, config=config)

    model.load_state_dict(torch.load(config.best_model_filename))
    test_acc_clean = evaluate_model(model, loader_clean, device, description="Final Test on Clean Dataset")
    test_acc_noisy1 = evaluate_model(model, loader_noise1, device, description=f"Final Test on Noisy Dataset (kernel={config.kernel_size1})")
    test_acc_noisy2 = evaluate_model(model, loader_noise2, device, description=f"Final Test on Noisy Dataset (kernel={config.kernel_size2})")
    wandb.run.summary["final_test_accuracy_clean"] = test_acc_clean
    wandb.run.summary["final_test_accuracy_kernel ={config.kernel_size1}"] = test_acc_noisy1
    wandb.run.summary["final_test_accuracy_kernel2 ={config.kernel_size2}"] = test_acc_noisy2
    # End the wandb run
    print("Training completed. Ending wandb run.")
    wandb.finish()
    
if __name__ == "__main__":
    probs = [0.5]
    group_norms = [0,8,16]
    unet_options = [False,True]
    for prob in probs:
        for group_norm in group_norms:
            for unet in unet_options:
                try:
                    main(prob, group_norm, unet)
                except Exception as e:
                    print(f"An error occurred during training with prob={prob}, group_norm={group_norm}, unet={unet}: {e}")
                    continue
