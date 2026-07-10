
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from base_training import *
if __name__ == "__main__":
    '''data_names = ["gtsrb", ]
    noise_types = ["gaussian"]
    for data_name in data_names:
        for noise_type in noise_types:
            main(prob=0., group_norm=0, Unet=False, data_name=data_name, noise_type=noise_type)
    '''
    data_names = ["imagenette","gtsrb"]
    noise_type = ["defocus_blur"]
    pretraineds = [True]
    for data_name in data_names:
        for noise in noise_type:
            probs = [0.5]
            group_norms = [0]
            unet_options = [True, False]
            for prob in probs:
                for group_norm in group_norms:
                    for unet in unet_options:
                        for pretrained in pretraineds:
                            main(prob, group_norm, unet, data_name, noise,model_name="Modifiedresnet18", pretrained=pretrained, train_method="method2")
                        
    
