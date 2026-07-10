import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from base_training import *

if __name__ == "__main__":
    '''data_names = ["gtsrb", "imagenette"]
    noise_types = ["gaussian", "motion_blur"]
    for data_name in data_names:
        for noise_type in noise_types:
            main(prob=0., group_norm=0, unet=False, data_name=data_name, noise_type=noise_type)
    '''
    data_names = ["gtsrb", "imagenette"]
    noise_type = ["defocus_blur"]
    pretraineds = [True, False]
    for data_name in data_names:
        for noise in noise_type:
            probs = [0.5]
            group_norms = [0,8]
            unet_options = [True,False]
            for prob in probs:
                for group_norm in group_norms:
                    for unet in unet_options:
                        for pretrained in pretraineds:
                            main(prob, group_norm, unet, data_name, noise, model_name="VIT", pretrained=pretrained)
        