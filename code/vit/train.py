from visualizer import*
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)

# 2. Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Unet import  UNetWrapper
from archtechre_common import *
import timm


def main(prob, group_norm, unet, data_name, noise_type, pretrained=False):

    entity_name = "wandb-mias-"  # Replace with your WandB entity name
    project_name = "Noise_Research"  # Replace with your WandB project name
    if prob == 0.5:
        target_run_name = "{}_{}_VIT_group_norm{}_Unet_{}".format(data_name, noise_type, group_norm, unet)
    else: 
        target_run_name = "{}_{}_VIT_prob{}_group_norm{}_Unet_{}".format(data_name, noise_type, prob, group_norm, unet)
    #target_run_name = "{}_{}_VIT_base_line".format(data_name, noise_type)
    if pretrained:
       target_run_name = f"{target_run_name}_pretrained"
    api = wandb.Api()
    runs = api.runs(path=f"{entity_name}/{project_name}", filters={"display_name": target_run_name},)
    found_run = False
    if len(runs) > 0:
        # An existing run was found! Grab its internal ID
        run_id = runs[0].id
        print(f"Found existing run! Resuming ID: {run_id}")
        found_run = True
    else:
        # No run found. Generate a fresh ID
        run_id = wandb.util.generate_id()
        print("No existing run found. Starting a new one.")
    if data_name == "imagenette":
        num_epochs = 5
    else:
        num_epochs = 5
    
    wandb.init(
    project=project_name,
    name=target_run_name,
    group="VIT",
    id=run_id,
    resume="allow",
    config={
        "learning_rate": 1e-4,
        "num_epochs": num_epochs,
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
        "kernel_size1": 31,
        "kernel_size2": 151,
        "best_model_filename": f"{target_run_name}.pth",
        #"best_model_filename": "{}_{}_VIT_base_line.pth".format(data_name, noise_type),
        "plot_every_n_epochs": 1,
        "group_norm_groups": group_norm,
        "UNet": unet,
        "data_name": data_name,
        "noise_type": noise_type,
        "pretrained": pretrained
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
        model = timm.create_model('vit_tiny_patch16_224',pretrained=True).to(device)
    else:
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
    if config.pretrained:

        print("Loading pretrained weights...")
        name = f"{config.data_name}_{config.noise_type}_VIT_base_line.pth"
        model.load_state_dict(torch.load(name))
    if config.group_norm_groups > 0:
        print(f"Replacing LayerNorm with GroupNorm (groups={config.group_norm_groups})...")
        model = replace_vit_layernorm_with_groupnorm(model, num_groups=config.group_norm_groups)
    if config.UNet:
        print("Wrapping the model with UNet...")
        model = UNetWrapper(base_model=model, in_channels=3, out_channels=3, base_features=16)
    if found_run:
        try:
            model.load_state_dict(torch.load(config.best_model_filename))
            print("Loaded model weights from previous run.")
        except FileNotFoundError:
            print("Model weights from previous run not found. Starting fresh.")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)
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
    '''data_names = ["gtsrb", "imagenette"]
    noise_types = ["gaussian", "motion_blur"]
    for data_name in data_names:
        for noise_type in noise_types:
            main(prob=0., group_norm=0, unet=False, data_name=data_name, noise_type=noise_type)
    '''
    data_names = ["gtsrb", "imagenette"]
    noise_type = ["gaussian", "motion_blur"]
    for data_name in data_names:
        for noise in noise_type:
            probs = [0.5]
            group_norms = [0,8]
            unet_options = [True,False]
            for prob in probs:
                for group_norm in group_norms:
                    for unet in unet_options:
                        main(prob, group_norm, unet, data_name, noise, pretrained=True)
        