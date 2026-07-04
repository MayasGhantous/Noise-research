import torch
import torchvision.models as models

def count_parameters(model):
    """Counts total and trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Load the models (without pre-trained weights since we just want to count parameters)
vgg11 = models.efficientnet_b4(weights=None)
vgg11_bn = models.vgg11_bn(weights=None)

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