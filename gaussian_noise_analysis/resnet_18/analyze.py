from pathlib import Path
import sys
from visualizer import *
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from archtechre_common import *
from Unet import UNetWrapper


def test_gaussian(model, loader_clean, loader_noise1, loader_noise2, device, std1=0.5, std2=1.0):
    clean_accuracy = evaluate_model(model, loader_clean, device, "Baseline (Clean Images)")
    noise1_accuracy = 0
    noise2_accuracy = 0
    for _ in range(5):  # Run multiple evaluations to average out randomness
        noise1_accuracy += evaluate_model(model, loader_noise1, device, f"Gaussian Noise (std={std1})")
        noise2_accuracy += evaluate_model(model, loader_noise2, device, f"Gaussian Noise (std={std2})")
    print(f"Average Accuracy for Clean Images: {clean_accuracy:.2f}%")
    print(f"Average Accuracy for Gaussian Noise (std={std1}): {noise1_accuracy / 5:.2f}%")
    print(f"Average Accuracy for Gaussian Noise (std={std2}): {noise2_accuracy / 5:.2f}%")


def analyze_batches(model, 
                    batch_clean_images, batch_clean_preds,
                    batch_noise1_images, batch_noise1_preds,
                    batch_noise2_images, batch_noise2_preds,device,index):
    """
    Takes 3 batches of images, their labels, and their pre-calculated predictions.
    Generates a comparison plot for every set of images at the same index.
    
    Returns a list of figures.
    """

    sample_images = []
    sample_tensors = []
    
    # ---------------------------------------------------------
    # Extract the i-th image from each batch
    # ---------------------------------------------------------
    img_clean = batch_clean_images[index]
    img_noise1 = batch_noise1_images[index]
    img_noise2 = batch_noise2_images[index]
    
    # Prepare images for visualization and tensors for the plotting function
    sample_images.append(denormalize(img_clean.cpu()).permute(1, 2, 0).numpy())
    sample_tensors.append(img_clean.unsqueeze(0).to(device))
    
    sample_images.append(denormalize(img_noise1.cpu()).permute(1, 2, 0).numpy())
    sample_tensors.append(img_noise1.unsqueeze(0).to(device))
    
    sample_images.append(denormalize(img_noise2.cpu()).permute(1, 2, 0).numpy())
    sample_tensors.append(img_noise2.unsqueeze(0).to(device))
    
    # ---------------------------------------------------------
    # Format the pre-calculated predictions for this index
    # ---------------------------------------------------------
    predicted_labels = [
        get_class_name(batch_clean_preds[index].item()),
        get_class_name(batch_noise1_preds[index].item()),
        get_class_name(batch_noise2_preds[index].item())
    ]
    
    # Optional: Format true labels if your plotting function needs them
    # true_labels = [
    #     get_class_name(batch_clean_labels[i].item()),
    #     get_class_name(batch_noise1_labels[i].item()),
    #     get_class_name(batch_noise2_labels[i].item())
    # ]
    
    # Generate the plot for this specific index
    # (Assuming display_multiple_images_progress still requires the 'model' argument)
    fig = display_multiple_images_progress(model, sample_tensors, sample_images, predicted_labels)
    
            
    return fig

def save_figures(model,visualizer,loader_clean, loader_noise1, loader_noise2,device, saving_location,max_samples=5):
    i = 0
    dictionary = {}
    
    flags = [True,False]

    for flag1 in flags:
            dictionary[flag1] = {}
            for flag2 in flags:
                dictionary[flag1][flag2] = {}
                for flag3 in flags:
                    dictionary[flag1][flag2][flag3] = {}
                    for labels in IMAGENETTE_CLASSES.keys():
                        dictionary[flag1][flag2][flag3][labels] = 0

    for batch_clean, batch_noise1, batch_noise2 in tqdm.tqdm(zip(loader_clean, loader_noise1, loader_noise2)):
        # 1. Unpack batches
        images_clean, labels_clean = batch_clean
        images_noise1, labels_noise1 = batch_noise1
        images_noise2, labels_noise2 = batch_noise2

        # 2. Move images to device for batched inference
        images_clean = images_clean.to(device)
        images_noise1 = images_noise1.to(device)
        images_noise2 = images_noise2.to(device)

        # 3. Calculate predictions for the ENTIRE batch at once
        model.eval()
        with torch.no_grad():
            preds_clean = torch.argmax(model(images_clean), dim=1)
            preds_noise1 = torch.argmax(model(images_noise1), dim=1)
            preds_noise2 = torch.argmax(model(images_noise2), dim=1)

        # 4. Pass everything into your plotting function
        
        
        for j in range(len(batch_clean[0])):  # Loop through each image in the batch
            flag1= preds_clean[j] == labels_clean[j].item()
            flag2 = preds_noise1[j] == labels_noise1[j].item()
            flag3 = preds_noise2[j] == labels_noise2[j].item()
            flag1 = flag1.item()
            flag2 = flag2.item()
            flag3 = flag3.item()
            if dictionary[flag1][flag2][flag3][labels_clean[j].item()] < max_samples:
                dictionary[flag1][flag2][flag3][labels_clean[j].item()] += 1
                save_path = Path(saving_location + f"/{flag1}_{flag2}_{flag3}/realLabel_{get_class_name(labels_clean[j].item())}")
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = str(save_path)
                fig = analyze_batches(
                    model,
                    images_clean, preds_clean,
                    images_noise1, preds_noise1,
                    images_noise2, preds_noise2,
                    device,j
                )
                fig.savefig(f"{save_path}/heatmap_{i}.png")
                plt.close(fig) 
                img_clean = batch_clean[0][j].squeeze(0).to(device)
                img_noisy = batch_noise1[0][j].squeeze(0).to(device)
                img_higher_order = batch_noise2[0][j].squeeze(0).to(device)
                true_label = labels_clean[j].item()

                # Generate the plot
                fig = visualizer.extract_and_return_figure(torch.stack([img_clean, img_noisy, img_higher_order]), [true_label, true_label, true_label])
                fig.savefig(f"{save_path}/feature_maps_{i}.png")
                plt.close(fig)  
            i += 1
def main_gaussian(model_name,models_location = str(Path(__file__).parent)+"/models"): 
    loader_clean, loader_noise1, loader_noise2 = get_test_loaders_for_gaussian(batch_size=32, std1=0.5, std2=1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.load_state_dict(torch.load(models_location+f"/{model_name}"))
    model = model.to(device)
    model.eval()
    if(isinstance(model, UNetWrapper)):
        model_visualizer = ResNet18FeatureVisualizer(model.get_base_model(), unet=model.get_unet())
    else:
        model_visualizer = ResNet18FeatureVisualizer(model)
    saving_location = str(Path(__file__).parent)+"/analysis_results/"+model_name
    #test_gaussian(model, loader_clean, loader_noise1, loader_noise2, device, std1=0.5, std2=1.0)
    save_figures(model, model_visualizer, loader_clean, loader_noise1, loader_noise2, device, saving_location, max_samples=5)
    # save_features(model,model_visualizer, loader_clean, loader_noise1, loader_noise2, device, saving_location)
if __name__ == "__main__":
    main_gaussian("base.pth")