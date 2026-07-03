import torch
import wandb
from visualizer import *


def main():
    wandb.init(
        project="Resnet-18",
        name="base1",
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
            "train_noise_prob": 0.,
            "eval_noise_std1": 0.5,
            "eval_noise_std2": 1.0,
            "best_model_filename": "base1.pth",
            "plot_every_n_epochs": 1,
            "group_norm_groups": 16,

            
        }
    )
    
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2 =get_traing_val_test_loaders_for_gaussian(config=config)
    print("Downloading/Loading pretrained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if config.group_norm_groups > 0:
        print(f"Replacing BatchNorm with GroupNorm (groups={config.group_norm_groups})...")
        model = replace_bn_with_gn(model, num_groups=config.group_norm_groups)
    '''model = models.resnet18()
    model.load_state_dict(torch.load("original.pth"))
    model = model.to(device)'''
    model = model.to(device)
    model_visualizer = ResNet18FeatureVisualizer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
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
    main()