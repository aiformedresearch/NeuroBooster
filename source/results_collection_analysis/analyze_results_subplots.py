import numpy as np
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

dpi = 500
dataset_name = 'AGE'

if dataset_name == 'ADNI':
    metrics = ['AUC', 'PR_AUC']
    bw_adjust = 0.7
    cut = 3
    source_folder_path_all_experiments = Path('/home/andreaespis/diciotti/andrea/med-booster/EXPERIMENTS_MRI_augm_21_11') / 'results_ADNI'
else:
    metrics = ['MAE', 'RMSE']
    bw_adjust = 0.9
    cut = 3 
    source_folder_path_all_experiments = Path('/home/andreaespis/diciotti/andrea/med-booster/EXPERIMENTS_MRI_augm_21_11') / 'results_AGE'
# Define the color mapping for each paradigm
paradigm_colors = {
    'SL': 'purple',
    'MB': 'lightblue',
    'VICReg': 'red',
    'SimMIM': 'orange'
}

matplotlib.rcParams.update({'font.size': 36})
seeds = range(30)
paradigms = ['supervised', 'neurobooster', 'vicreg', 'simim']
folds = range(1)
labels_percentage_list = [100, 10, 1]

destination_folder_path = source_folder_path_all_experiments

results_mean_file = open(destination_folder_path / "results_mean.txt", "a", buffering=1)

fig, axs = plt.subplots(len(labels_percentage_list), len(metrics), figsize=(22, 22))  # Create subplots for each label percentage and metric

for row_idx, labels_percentage in enumerate(labels_percentage_list):
    for col_idx, metric in enumerate(metrics):
        df = pd.read_csv(source_folder_path_all_experiments / f'results_collected_{metric}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv')
        paradigms_names = {'supervised': 'SL', 'neurobooster': 'MB', 'vicreg': 'VICReg', 'simim': 'SimMIM', 'baseline': 'baseline'}
        df = df.drop('Unnamed: 0', axis=1, errors='ignore')
        df.columns = [paradigms_names[col] for col in df.columns]
        df_results = df


        print('Plotting distributions')
        plt.rcParams["figure.figsize"] = (10, 2)

        # Plot all KDEs together to ensure consistent y-axis normalization
        combined_data = pd.melt(df_results, var_name='Paradigm', value_name='Value')
        ax = sns.kdeplot(data=combined_data, x='Value', hue='Paradigm', fill=True, common_norm=True, bw_adjust=bw_adjust, cut=cut, ax=axs[row_idx, col_idx], palette=paradigm_colors, legend=False)

        # Add rug plots separately for each paradigm
        for paradigm in df_results.columns:
            color = paradigm_colors.get(paradigm, 'grey')
            sns.rugplot(data=df_results[paradigm], height=0.1, ax=axs[row_idx, col_idx], color=color)
        
        if metric == 'AUC':
            axs[row_idx, col_idx].set_xlabel('ROC-AUC')
        elif metric == 'PR_AUC':
            axs[row_idx, col_idx].set_xlabel('PR-AUC')
        elif metric == 'MAE':
            axs[row_idx, col_idx].set_xlabel('MAE (years)')
        elif metric == 'RMSE':
            axs[row_idx, col_idx].set_xlabel('RMSE (years)')

        
        axs[row_idx, col_idx].set_ylabel('PDE')

        # Adjust y-axis limits to better fit the distributions
        y_data = []
        for line in ax.get_lines():
            y_data.extend(line.get_ydata())
        if y_data:
            y_max = max(y_data)
            axs[row_idx, col_idx].set_ylim(0, y_max * 1.1)

        # Remove the box around each plot except the left spine (y-axis line)
        axs[row_idx, col_idx].spines['top'].set_visible(False)
        axs[row_idx, col_idx].spines['right'].set_visible(False)
        axs[row_idx, col_idx].spines['bottom'].set_visible(True)
        axs[row_idx, col_idx].spines['left'].set_visible(True)  # Make the y-axis line visible

        print(f'\n{metric}, {labels_percentage}', file=results_mean_file)
        print(df_results.iloc[:, 1:].mean(), file=results_mean_file)

# Add titles for each row
row_titles = ['Labels percentage 100%', 'Labels percentage 10%', 'Labels percentage 1%']
for row_idx, row_title in enumerate(row_titles):
    fig.text(0.5, 0.92 - row_idx * 0.305, row_title, ha='center', va='bottom', fontsize=36)

# Add a custom legend on top
legend_handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in paradigm_colors.items()]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=len(paradigm_colors), title='Paradigm')

# Adjust layout to prevent overlapping and reduce white space
plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.90, bottom=0.1, left=0.08, right=0.95)  # Increase space between rows and columns

# Save figure with all plots
plt.savefig(destination_folder_path / f'new_center_title_combined_plots_AUC_PRAUC_{bw_adjust}_{cut}_{dpi}dpi.png', dpi=dpi, bbox_inches='tight')
plt.close()

results_mean_file.close()