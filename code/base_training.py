from resnet_18.visualizer import *
from vit.visualizer import *
from Unet import  UNetWrapper
from torchvision import models
import timm
from resnet_18_with_no_skip_connections.network import create_resnet18_without_skip

def main(prob,group_norm,unet,data_name,noise_type,model_name, pretrained=False, train_method="method2"):
    entity_name = "wandb-mias-"  # Replace with your WandB entity name
    project_name = "Noise_Research"  # Replace with your WandB project name
    if prob == 0.5:
        if model_name == "resnet18":
            target_run_name = f"{data_name}_{noise_type}_resnet18_group_norm{group_norm}_Unet_{unet}"
        elif model_name == "Modifiedresnet18":
            target_run_name = f"{data_name}_{noise_type}_Modifiedresnet18_group_norm{group_norm}_Unet_{unet}"
        elif model_name == "VIT":
            target_run_name = f"{data_name}_{noise_type}_VIT_group_norm{group_norm}_Unet_{unet}"
    else:
        if model_name == "resnet18":
            target_run_name = f"{data_name}_{noise_type}_resnet18_prob{prob}_group_norm{group_norm}_Unet_{unet}"
        elif model_name == "Modifiedresnet18":
            target_run_name = f"{data_name}_{noise_type}_Modifiedresnet18_prob{prob}_group_norm{group_norm}_Unet_{unet}"
        elif model_name == "VIT":
            target_run_name = f"{data_name}_{noise_type}_VIT_prob{prob}_group_norm{group_norm}_Unet_{unet}"
    target_run_name = f"{target_run_name}_{train_method}"
    #target_run_name = f"{data_name}_{noise_type}_resnet18_base_line"
    if pretrained:
        target_run_name = f"{target_run_name}_pretrained"
    api = wandb.Api()
    runs = api.runs(path=f"{entity_name}/{project_name}", filters={"display_name": target_run_name})
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
    if model_name == "resnet18":
        group_name = "Resnet_18"
    elif model_name == "Modifiedresnet18":
        group_name = "ModifiedResnet18"
    elif model_name == "VIT":
        group_name = "VIT"
    wandb.init(
        project=project_name,
        group=group_name,
        name=target_run_name,
        id=run_id,
        resume="allow",  # This allows resuming if the run ID already exists
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
            "kernel_size1": 51,
            "kernel_size2": 91,
            "radius1": 10,
            "radius2": 25,
            "best_model_filename": f"{target_run_name}.pth",
            #"best_model_filename": f"{data_name}_{noise_type}_resnet18_base_line.pth",
            "plot_every_n_epochs": 1,
            "group_norm_groups": group_norm,
            "UNet": unet,
            "data_name": data_name,
            "noise_type": noise_type,
            "pretrained": pretrained,
            "train_method": train_method
        }
    )
    
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if config.train_method == "method2":
        train_loader, train_loader2, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2 = get_traing_val_test_loaders(config)
    elif config.train_method == "method1":
        train_loader, val_loader, val_loader2, val_loader3, loader_clean, loader_noise1, loader_noise2 = get_traing_val_test_loaders(config)    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "Modifiedresnet18":
        model = create_resnet18_without_skip()
    elif model_name == "VIT":
        print("Downloading/Loading pretrained VIT...")
        if config.data_name == "gtsrb":
            model = timm.create_model('vit_tiny_patch16_224',pretrained=True).to(device)
        else:
            model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
    
    if config.pretrained:
        print("Loading pretrained weights...")
        try:
            if model_name == "resnet18":
                name = f'resnet_18/{config.data_name}_resenet18_pretrained.pth'
            elif model_name == "Modifiedresnet18":
                name = f'resnet_18_with_no_skip_connections/{config.data_name}_Modifiedresnet18_pretrained.pth'
            elif model_name == "VIT":
                name = f'vit/{config.data_name}_VIT_pretrained.pth'
            model.load_state_dict(torch.load(name))
        except FileNotFoundError:
            print("Pretrained weights not found.")
            raise FileNotFoundError(f"Pretrained weights not found at {name}. Please ensure the file exists.")
    
    if config.group_norm_groups > 0:
        print(f"Replacing BatchNorm with GroupNorm (groups={config.group_norm_groups})...")
        if model_name == "VIT":
            print(f"Replacing LayerNorm with GroupNorm (groups={config.group_norm_groups})...")
            model = replace_vit_layernorm_with_groupnorm(model, num_groups=config.group_norm_groups)
        else:
            model = replace_bn_with_gn(model, num_groups=config.group_norm_groups)
    if config.UNet:
        print("Wrapping the model with UNet...")
        model = UNetWrapper(base_model=model)
    if found_run:
        try:
            model.load_state_dict(torch.load(config.best_model_filename))
            print("Loaded model weights from previous run.")
        except FileNotFoundError:
            print("Model weights from previous run not found. Starting fresh.")
    model = model.to(device)
    
    if config.UNet:
        if model_name == "VIT":
            model_visualizer = ViTBatchAttentionVisualizer(model.get_base_model(), unet=model.get_unet())
        else:
            model_visualizer = ResNet18FeatureVisualizer(model.get_base_model(), unet=model.get_unet())
    else:
        if model_name == "VIT":
            model_visualizer = ViTBatchAttentionVisualizer(model)
        else:
            model_visualizer = ResNet18FeatureVisualizer(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    if config.train_method == "method1":
        train_model(model, train_loader, val_loader, val_loader2, val_loader3, criterion, optimizer, device,prog_vis =model_visualizer, config=config)
    elif config.train_method == "method2":
        train_model2(model, train_loader,train_loader2, val_loader, val_loader2, val_loader3, criterion, optimizer, device,prog_vis =model_visualizer, config=config)
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