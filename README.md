## Environments
We use 2 environments. The first one is named `requirements.txt`, where regular `opencv-python` is included; these are for regular operating systems with a GUI.<br>
The other one is `requirements_runpods.txt` that has `opencv-python-headless`, where the operating system does not have a GUI.<br>
## Training: 
#### 
To run the code training, open the relevant `train.py` (e.g., under `code/`) and use the main function from `base_training.py`.<br>
### Parameters of This Main: 
prob (float, 0 to 1): percentage of the dataset that is noisy. If 0, there is no noisy data in the dataset.<br><br>
group_norm (int): if > 0, replaces normalization layers in the architecture with GroupNorm using `group_norm` groups; if 0, the default normalization is used.
<br>

unet (boolean): Whether to wrap the model in a U-Net or not.
<br>

data_name (string): Should be "imagenette" or "gtsrb" depending on which dataset you want to use.
<br>

noise_type(string): Either "gaussian","motion_blur" or "defocus_blur".
<br>

model_name(string): Is the model type you need to train. It should be "resnet18","Modifiedresnet18" or "VIT".
<br>

pretrained(boolean): If you want to use a model that was pretrained on the clean dataset.
<br>

train_method(string): Should be "method1" or "method2", where method1 is the regular CrossEntropyLoss and for method2 we added another loss and an MSE loss that makes the Unet output for the noisy image and the regular the same. 
<br> 

Notes: 
1.  If you want to use pretraining then you must have first train th model with prob = 0 , unet = false, grop_norm = 0  and method1, this would save a pretrained model on the model directory.
2. If you want to change another hyper parameters then you should open base_traning,py there exists all of the other hyper paramters.
3. You must connect to a wandb account so you can see the visulization online while tranining.
4. If for some reason the training stops, the code handles it just rerun the same command.

## Analysis
After training, each model directory should include `.pth` files. We have two functions for analysis.<br><br>
The first is `main`, which saves plots for each model output combination on the image. For example, if the model gets the clean image and noise1 image right, but not noise2, the combination is `true_true_false`.<br><br>
If the true label is `english_springer`, the code saves up to 5 results for `true_true_false` with that true label, and writes the remaining results to a JSON file so you can check each image result later if needed.
<br><br>
The other function is `save_figure_for_index`. It takes almost the same parameters as `main` in training, and also gets the model locations, the result-saving location, and the load function (which every model has in `analyze.py`). This function is implemented to compare how different models handle the same image and noise.
<br><br>
To make this easier, each `analyze.py` file has implemented lists that are zipped together so we can run the code for multiple models at the same time.