import types
import torchvision.models as models
from torchvision.models.resnet import BasicBlock

# 1. Define a new forward function without the skip connection
def forward_without_skip(self, x):
    # Pass through the first convolutional block
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    # Pass through the second convolutional block
    out = self.conv2(out)
    out = self.bn2(out)

    out = self.relu(out)
    return out

def create_resnet18_without_skip():
    # 2. Load standard ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Use weights=None for training from scratch

    # 3. Overwrite the forward method for all BasicBlocks
    for module in model.modules():
        if isinstance(module, BasicBlock):
            # Bind the custom function to this specific module instance
            module.forward = types.MethodType(forward_without_skip, module)
    return model