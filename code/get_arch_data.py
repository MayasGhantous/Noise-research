import timm
from torchvision import models
from torchvision.datasets import Imagenette as TVImagenette, GTSRB as TVGTSRB
import os
from pathlib import Path
PARENT_DIR = str(Path(__file__).parent.parent)
DATA_DIR = os.path.join(PARENT_DIR, "data")
print(timm.create_model('vit_tiny_patch16_224',pretrained=True))
print(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1))

imagenette_train = TVImagenette(root=DATA_DIR, split='train', size='160px', download=True)
print(f"imnet train size: {len(imagenette_train)}")

imagenette_test = TVImagenette(root=DATA_DIR, split='val', size='160px', download=True)
print(f"imnet test size: {len(imagenette_test)}")

gtsrb_train = TVGTSRB(root=DATA_DIR, split='train', download=True)
print(f"gtsrb train size: {len(gtsrb_train)}")

gtsrb_test = TVGTSRB(root=DATA_DIR, split='test', download=True)
print(f"gtsrb test size: {len(gtsrb_test)}")

