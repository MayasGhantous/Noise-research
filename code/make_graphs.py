import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
import ast

def make_comparison(models):
    df = pd.read_csv('csv_files/project.csv')

    metrics = [
        'final_test_accuracy_clean', 
        'final_test_accuracy_noisy1', 
        'final_test_accuracy_noisy2'
    ]

    # Parse the JSON-like text in the summary column
    def parse_summary(row):
        try:
            # If the string uses single quotes, use ast.literal_eval(row) instead of json.loads(row)
            parsed = ast.literal_eval(row)
            return pd.Series([parsed.get(m) for m in metrics])
        except:
            return pd.Series([None, None, None])

    df[metrics] = df['summary'].apply(parse_summary)
    df['name'] = df['name'].str.replace("_method1", "", regex=False)
    df['base_name'] = df['name'].str.replace(models[1], models[0], regex=False)

    # for if models[0] is resnet18 and models[1] is Modifiedresnet18
    df_resnet = df[df['name'].str.contains(models[0], regex=False) & ~df['name'].str.contains(models[1], regex=False)]
    df_modified = df[df['name'].str.contains(models[1], regex=False)]

    merged = pd.merge(
        df_resnet, 
        df_modified, 
        left_on='name', 
        right_on='base_name', 
        suffixes=("_"+models[0], "_"+models[1])
    )

    output_cols = [f'name_{models[0]}', f'name_{models[1]}']
    for metric in metrics:
        diff_col = f'{metric}_diff'
        merged[diff_col] = merged[f'{metric}_{models[1]}'] - merged[f'{metric}_{models[0]}']
        output_cols.append(diff_col)

    result = merged[output_cols]
    result.to_csv(f'csv_files/metric_differences_{models[0]}_vs_{models[1]}.csv', index=False)

def make_graphs():
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

make_comparison(["Unet_False","Unet_True"])
