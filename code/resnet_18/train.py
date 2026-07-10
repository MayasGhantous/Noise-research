import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from base_training import *

if __name__ == "__main__":
    data_names = ["imagenette"]
    noise_types = ["defocus_blur","motion_blur","gaussian"]
    pretraineds = [True]
    for data_name in data_names:
        for noise in noise_types:
            probs = [0.5]
            group_norms = [0,8]
            unet_options = [True]
            for prob in probs:
                for group_norm in group_norms:
                    for unet in unet_options:
                        for pretrained in pretraineds:
                            main(prob, group_norm, unet, data_name, noise,model_name = "resnet18", pretrained=pretrained, train_method="method2")
    