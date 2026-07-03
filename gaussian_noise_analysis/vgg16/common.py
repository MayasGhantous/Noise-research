import torch
from torchvision import models
import sys
from pathlib import Path


parent_dir = str(Path(__file__).parent.parent)

# 2. Add the parent directory to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from archtechre_common import *

# --- Custom Download Function ---
# --- Main Script ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    std1 = 0.5
    std2 = 1.0
    loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_gaussian(batch_size=32, std1=std1, std2=std2)
    print("Downloading/Loading pretrained VGG16...")
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    model = model.to(device)
    evaluate_model(model, loader_clean, device, "Baseline (Clean Images)")
    evaluate_model(model, loader_noise1, device, "Gaussian Noise (std=0.5)")
    evaluate_model(model, loader_noise2, device, "Gaussian Noise (std=1.0)")

if __name__ == "__main__":
    main()