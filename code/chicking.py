
import torch
import torchvision.models as models
import Unet
from archtechre_common import *
import matplotlib.pyplot as plt
from resnet_18.visualizer import ResNet18FeatureVisualizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_loader, val_loader2, val_loader3 = get_test_loaders_for_motion_blur(batch_size=25, kernel_size1=71, kernel_size2=71, data_name="gtsrb")
rand_idx = 690
img, true_label = val_loader.dataset[rand_idx]
img2, _ = val_loader2.dataset[rand_idx]
img3, _ = val_loader3.dataset[rand_idx]

img_clean = img.squeeze(0).to(device)
img_noisy = img2.squeeze(0).to(device)
img_higher_order = img3.squeeze(0).to(device)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
visualizer = ResNet18FeatureVisualizer(model)

# Generate the plot
fig = visualizer.extract_and_return_figure(IMAGENETTE,torch.stack([img_clean, img_noisy, img_higher_order]), [true_label, true_label, true_label])
plt.show()


