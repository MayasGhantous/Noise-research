## Environments
We use 2 environments. The first one is named `requirements.txt`, where regular `opencv-python` is included; these are for regular operating systems with a GUI.<br>
The other one is `requirements_runpods.txt` that has `opencv-python-headless`, where the operating system does not have a GUI.<br>
## training: 
#### 
to run the code training, open the relevant `train.py` (e.g., under `code/`) and use the main function from `base_training.py`.<br>
### the parameters of this main: 
prob (float, 0 to 1): percentage of the dataset that is noisy. If 0, there is no noisy data in the dataset.<br><br>
group_norm (int): if > 0, replaces normalization layers in the architecture with GroupNorm using `group_norm` groups; if 0, the default normalization is used.
<br>
unet(boolean): do I want to rap the model in a unet or not
<br>
<br>
data_name(string): is should be "imagenette" or "gtsrb" depends what data set you want to use.

<br><br>
noise_type(string): its eather "gaussian","motion_blur" or "defocus_blur".
<br><br>
model_name(string): is the model type you need to train it should be "resnet18","Modifiedresnet18" or "VIT".
<br><br>
pretrained(boolean): means if you want to use a pretrained model on the clean dataset 
<br><br>
train_method(string): it should be "method1" or "method2", where method1 is the regular CrossEntropyLoss and for method2 we added another loss and an MSE loss that makes the Unet output for the noisy image and the regular the same. 
<br> <br>
notes: 
1.  if you want to use pretraining then you must have first train th model with prob = 0 , unet = false, grop_norm = 0  and method1, this would save a pretrained model on the model directory.
2. if you want to change another hyper parameters then you should open base_traning,py there exists all of the other hyper paramters.
3. you must connect to a wandb account so you can see the visulization online while tranining.
4. if for some reason the training stops the code handles it just rerun the same command.

## Analysis
after traning, each model's directory must have some models that ends with pth, we have 2 function for analyzation.<br><br>
 first is main which will save plots for each compantion of the model reasults on the image, for example if the model got the clean image and noise1 image and did not ge the noise 2 this compenation is true_true_false. <br><br>
  if the true label is english_springer for example, the code saves at max 5 resutls for true_true_false and the true lable is english spriger, and saves the other resutls on a jason file so we can chick what is the reults of each image so we can see it later if needed. 
  <br><br>
the other funciton is called save_figure_for_index it takes almost the same paramters as the main in traning but also get teh models location the saveing locaiton od the results and the load funciton (which every model has in the analyze.py), this function has been implemnted to see how diffrent models cope with the same image and noise.
<br><br>
to make it easer in each analyze.py file we have the implemnted listst and ziped them toghter so we can run the code for number of models at the same time.