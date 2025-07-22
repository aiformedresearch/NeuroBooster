import numpy as np
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

dpi = 500
dataset_name = 'AGE'  # Change to 'ADNI' if needed

if dataset_name == 'ADNI':
    metrics = ['AUC', 'PR_AUC']
    bw_adjust = 0.7
    cut = 5
    source_folder_path_all_experiments = Path('/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS') / 'results_definitive_ADNI'
else:
    metrics = ['MAE', 'RMSE']
    bw_adjust = 0.9
    cut = 5
    source_folder_path_all_experiments = Path('/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS') / 'results_definitive_AGE'

# Color-blind-safe color and linestyle mappings
paradigm_colors = {
    'SL': '#008000',         # Green
    'NB': '#1E90FF',         # Dodger Blue
    'SimCLR': '#FF69B4',     # Hot pink
    'VICReg': '#C71585',     # Dark pink
    'MAE': '#FFA500',        # Orange
    'SimMIM': '#FF8C00',     # Dark orange
}

paradigm_linestyles = {
    'SL': '-',
    'NB': '-',
    'SimCLR': '-',
    'VICReg': '--',
    'MAE': '-',
    'SimMIM': '--',
}

matplotlib.rcParams.update({'font.size': 36})
seeds = range(23)
paradigms = ['supervised', 'neurobooster', 'simclr', 'vicreg', 'mae', 'simim']
folds = range(1)
labels_percentage_list = [100, 10, 1]

destination_folder_path = source_folder_path_all_experiments
results_mean_file = open(destination_folder_path / "results_mean.txt", "a", buffering=1)

fig, axs = plt.subplots(len(labels_percentage_list), len(metrics), figsize=(22, 22))

for row_idx, labels_percentage in enumerate(labels_percentage_list):
    for col_idx, metric in enumerate(metrics):
        df = pd.read_csv(source_folder_path_all_experiments / f'results_collected_{metric}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv')
        paradigms_names = {'supervised': 'SL', 'neurobooster': 'NB', 'vicreg': 'VICReg', 'simim': 'SimMIM', 'baseline': 'baseline', 'mae': 'MAE', 'simclr': 'SimCLR'}
        df = df.drop('Unnamed: 0', axis=1, errors='ignore')
        df.columns = [paradigms_names[col] for col in df.columns]
        df_results = df

        print('Plotting distributions')
        plt.rcParams["figure.figsize"] = (10, 2)

        combined_data = pd.melt(df_results, var_name='Paradigm', value_name='Value')
        x_min = combined_data['Value'].min()
        x_max = combined_data['Value'].max()

        for paradigm in df_results.columns:
            color = paradigm_colors.get(paradigm, 'grey')
            linestyle = paradigm_linestyles.get(paradigm, '-')
            sns.kdeplot(
                data=combined_data[combined_data['Paradigm'] == paradigm],
                x='Value',
                ax=axs[row_idx, col_idx],
                fill=True,
                bw_adjust=bw_adjust,
                cut=cut,
                color=color,
                linewidth=2,
                linestyle=linestyle,
                label=paradigm,
                clip=(x_min - 1, x_max + 1)
            )
            sns.rugplot(
                data=df_results[paradigm],
                height=0.05,
                ax=axs[row_idx, col_idx],
                color=color,
                clip_on=False
            )

        axs[row_idx, col_idx].axhline(0, color='black', linewidth=0.5, linestyle='--')

        # Axis labels
        if metric == 'AUC':
            axs[row_idx, col_idx].set_xlabel('ROC-AUC')
        elif metric == 'PR_AUC':
            axs[row_idx, col_idx].set_xlabel('PR-AUC')
        elif metric == 'MAE':
            axs[row_idx, col_idx].set_xlabel('MAE (years)')
        elif metric == 'RMSE':
            axs[row_idx, col_idx].set_xlabel('RMSE (years)')

        axs[row_idx, col_idx].set_ylabel('PDE')

        # X-axis limit
        axs[row_idx, col_idx].set_xlim(x_min - 1, x_max + 1)

        # Y-axis: custom dynamic scaling for ADNI only
        if dataset_name == 'ADNI':
            y_max = 0
            for line in axs[row_idx, col_idx].get_lines():
                y_data = line.get_ydata()
                if len(y_data) > 0:
                    y_max = max(y_max, max(y_data))
            boosted_y_max = max(y_max * 4, 0.04)  # Stronger boost
            axs[row_idx, col_idx].set_ylim(0, boosted_y_max)
        else:
            axs[row_idx, col_idx].set_ylim(0, None)

        axs[row_idx, col_idx].spines['top'].set_visible(False)
        axs[row_idx, col_idx].spines['right'].set_visible(False)
        axs[row_idx, col_idx].spines['bottom'].set_visible(True)
        axs[row_idx, col_idx].spines['left'].set_visible(True)

        print(f'\n{metric}, {labels_percentage}', file=results_mean_file)
        print(df_results.iloc[:, 1:].mean(), file=results_mean_file)

# Row titles
row_titles = ['Label percentage 100%', 'Label percentage 10%', 'Label percentage 1%']
for row_idx, row_title in enumerate(row_titles):
    fig.text(0.5, 0.91 - row_idx * 0.30, row_title, ha='center', va='bottom', fontsize=36)

# Custom legend
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=paradigm_colors[p], linestyle=paradigm_linestyles[p], linewidth=4, label=p)
    for p in ['SL', 'NB', 'SimCLR', 'VICReg', 'MAE', 'SimMIM']
]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=3, title='Paradigm')

plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.90, bottom=0.1, left=0.08, right=0.95)
plt.savefig(destination_folder_path / f'new_center_title_combined_plots_{metrics}_{bw_adjust}_{cut}_{dpi}dpi.png', dpi=dpi, bbox_inches='tight')
plt.close()

results_mean_file.close()
