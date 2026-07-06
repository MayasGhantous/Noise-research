from visualizer import*
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)

# 2. Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Unet import  UNetWrapper


def main(prob, group_norm, unet, data_name, noise_type):
    wandb.init(
    project="Noise-Research",
    name="{}_VIT_group_norm{}_Unet_{}".format(noise_type, group_norm, unet),
    config={
        "learning_rate": 1e-4,
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
        "best_model_filename": "{}_{}_VIT_prob{}_group_norm{}.pth".format(data_name, noise_type, prob, group_norm),
        "plot_every_n_epochs": 1,
        "group_norm_groups": group_norm,
        "UNet": unet,
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
    print("Downloading/Loading pretrained VIT...")
    if config.data_name == "gtsrb":
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False).to(device)
    else:
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
    if config.group_norm_groups > 0:
        print(f"Replacing LayerNorm with GroupNorm (groups={config.group_norm_groups})...")
        model = replace_vit_layernorm_with_groupnorm(model, num_groups=config.group_norm_groups)
    if config.UNet:
        print("Wrapping the model with UNet...")
        model = UNetWrapper(base_model=model, in_channels=3, out_channels=3, base_features=16)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    if config.UNet:
        model_visualizer = ViTBatchAttentionVisualizer(model.get_base_model(), unet=model.get_unet())
    else:
        model_visualizer = ViTBatchAttentionVisualizer(model)
    # 6. Train and finish
    train_model(model, train_loader, val_loader, val_loader2, val_loader3, criterion, optimizer, device,prog_vis =model_visualizer, config=config)

    model.load_state_dict(torch.load(config.best_model_filename))
    test_acc_clean = evaluate_model(model, loader_clean, device, description="Final Test on Clean Dataset")
    test_acc_noisy1 = evaluate_model(model, loader_noise1, device, description=f"Final Test on Noisy Dataset")
    test_acc_noisy2 = evaluate_model(model, loader_noise2, device, description=f"Final Test on Noisy Dataset")
    wandb.run.summary["final_test_accuracy_clean"] = test_acc_clean
    wandb.run.summary["final_test_accuracy_noisy1"] = test_acc_noisy1
    wandb.run.summary["final_test_accuracy_noisy2"] = test_acc_noisy2
    # End the wandb run
    print("Training completed. Ending wandb run.")
    wandb.finish()

if __name__ == "__main__":
    probs = [0.5]
    for prob in probs:
        main(prob=prob, group_norm=0, unet=False)  # You can change the probability and group_norm values as needed