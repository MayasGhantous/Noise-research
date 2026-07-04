import torch
import torchvision.models as models
import Unet

def count_parameters(model):
    """Counts total and trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Load the models (without pre-trained weights since we just want to count parameters)
vgg11 = models.resnet18(weights=None)
vgg11_bn = Unet.UNetPreProcessor(base_model=models.resnet18(weights=None), in_channels=3, out_channels=3, base_features=16)

# Count parameters for VGG11
vgg11_total, vgg11_trainable = count_parameters(vgg11)

# Count parameters for VGG11-BN
vgg11_bn_total, vgg11_bn_trainable = count_parameters(vgg11_bn)

# Print the results
print("=== VGG11 ===")
print(f"Total Parameters:     {vgg11_total:,}")
print(f"Trainable Parameters: {vgg11_trainable:,}\n")

print("=== VGG11-BN ===")
print(f"Total Parameters:     {vgg11_bn_total:,}")
print(f"Trainable Parameters: {vgg11_bn_trainable:,}")