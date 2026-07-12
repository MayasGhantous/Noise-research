import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CSV_PATH = "csv_files"
PLT_PATH = "plots"
file_names = ["vit_clean_val", "vit_noise1_val", "vit_noise2_val"]
df_arr = []
for i in range(3):
    df_arr.append(pd.read_csv(CSV_PATH + "/" + file_names[i] + ".csv"))

fig, axd = plt.subplot_mosaic([["A", "B"], ["C", "C"]], figsize=(8, 8))
axd["C"].set_box_aspect(1)
all_col_names = [
    ["gtsrb_gaussian_VIT_group_norm0_Unet_True - Clean Validation Accuracy", "gtsrb_gaussian_VIT_group_norm0_Unet_False - Clean Validation Accuracy"],
    ["gtsrb_gaussian_VIT_group_norm0_Unet_True - Noisy Validation Accuracy",
     "gtsrb_gaussian_VIT_group_norm0_Unet_False - Noisy Validation Accuracy"],
    ["gtsrb_gaussian_VIT_group_norm0_Unet_True - Higher Order Validation Accuracy",
     "gtsrb_gaussian_VIT_group_norm0_Unet_False - Higher Order Validation Accuracy"]]
# Plot the specified columns for each file across the side-by-side subplots
for ax, df, title, col_names in zip(
    axd.values(), [df_arr[0], df_arr[1], df_arr[2]], 
    ["Clean Validation", "Gaussian Noise Validation", "High Gaussian Noise Validation"],
    all_col_names):
    print(df)
    ax.plot(df.index + 1, df[col_names])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(["With U-Net", "Without U-Net"])
    ax.grid(True)

plt.tight_layout()
plt.savefig(PLT_PATH+"/ViT_Unet_plot")
plt.show()


