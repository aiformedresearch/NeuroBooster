import os
import pandas as pd
import numpy as np

# Path to your results folder
folder = '/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_07_02_LARS_long_no_augm_resnet34/results_definitive'

# List all .csv files in the folder
for filename in os.listdir(folder):
    if filename.endswith('.csv') and not filename.endswith('_mean_std.csv'):
        filepath = os.path.join(folder, filename)
        df = pd.read_csv(filepath, index_col=0)

        # Calculate mean and std
        mean_series = df.mean(axis=0)
        std_series = df.std(axis=0)

        # Format each value as "x (y)" with two decimals
        summary = {col: f"{mean_series[col]:.2f} ({std_series[col]:.2f})" for col in df.columns}

        # Create new DataFrame and save it
        result_df = pd.DataFrame([summary])
        output_filename = filename.replace('.csv', '_mean_std.csv')
        output_path = os.path.join(folder, output_filename)
        result_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
