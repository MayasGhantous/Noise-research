import os
from pathlib import Path
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm 
import wandb

from torchvision import transforms
from torchvision.datasets import Imagenette as TVImagenette, GTSRB as TVGTSRB
from torch.utils.data import DataLoader, Subset

from archtechre_common import *

base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    
normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_clean = transforms.Compose([*base_transforms])
transform_defocus_blur1 = transforms.Compose([*base_transforms, AddDefocusBlur(severity=5)])
transform_defocus_blur2 = transforms.Compose([*base_transforms, AddDefocusBlur(severity=6)])

dataset_blurred = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_defocus_blur1)
dataset_very_blurred = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_defocus_blur2)
dataset_clean = TVGTSRB(root=DATA_DIR, split='test', download=True, transform=transform_clean)

#cv2.imshow("Blurred Sample", dataset_blurred[0][0].numpy())
sample = 30
cv2.imshow(f"Blurred Sample {sample}", dataset_blurred[sample][0].permute(1, 2, 0).numpy())
cv2.imshow(f"Very Blurred Sample {sample}", dataset_very_blurred[sample][0].permute(1, 2, 0).numpy())
cv2.imshow(f"Clean Sample {sample}", dataset_clean[sample][0].permute(1, 2, 0).numpy())

sample = 18
cv2.imshow(f"Blurred Sample {sample}", dataset_blurred[sample][0].permute(1, 2, 0).numpy())
cv2.imshow(f"Very Blurred Sample {sample}", dataset_very_blurred[sample][0].permute(1, 2, 0).numpy())
cv2.imshow(f"Clean Sample {sample}", dataset_clean[sample][0].permute(1, 2, 0).numpy())


cv2.waitKey(0)