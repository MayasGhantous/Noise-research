import wandb
import torch
import torch.nn as nn
from torchvision import models
import sys
from pathlib import Path
from visualizer import*

parent_dir = str(Path(__file__).parent.parent)

# 2. Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from archtechre_common import *

def main():
    wandb.init(
    project="vgg16_training",
    name="base_no_noisy_training",
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
        "train_noise_prob": 0.5,
        "eval_noise_std1": 0.5,
        "eval_noise_std2": 1.0,
        "best_model_filename": "base_no_noisy_training.pth",
        "plot_every_n_epochs": 1,
        "group_norm_groups": 16,
    }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2 =get_traing_val_test_loaders_for_gaussian(config=config)
    print("Downloading/Loading pretrained VGG16...")
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model_visualizer = VGG16FeatureVisualizer(model)
    # 6. Train and finish
    train_model(model, train_loader, val_loader, val_loader2, val_loader3, criterion, optimizer, device,prog_vis =model_visualizer, config=config)

    model.load_state_dict(torch.load(config.best_model_filename))
    test_acc_clean = evaluate_model(model, loader_clean, device, description="Final Test on Clean Dataset")
    test_acc_noisy1 = evaluate_model(model, loader_noise1, device, description=f"Final Test on Noisy Dataset (std={config.eval_noise_std1})")
    test_acc_noisy2 = evaluate_model(model, loader_noise2, device, description=f"Final Test on Noisy Dataset (std={config.eval_noise_std2})")
    wandb.run.summary["final_test_accuracy_clean"] = test_acc_clean
    wandb.run.summary["final_test_accuracy_noisy1 std=1.0"] = test_acc_noisy1
    wandb.run.summary["final_test_accuracy_noisy2 std=2.0"] = test_acc_noisy2
    # End the wandb run
    print("Training completed. Ending wandb run.")
    wandb.finish()
    

if __name__ == "__main__":
    main()