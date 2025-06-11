import numpy as np
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

dpi = 500
dataset_name = 'AGE'

metrics = ['MAE']
bw_adjust = 0.9
cut = 3
source_folder_path_all_experiments = Path('/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS') / 'results_AGE'

paradigm_colors = {
    'SL': 'purple',
    'MB': 'lightblue',
    'VICReg': 'red',
    'SimMIM': 'orange',
    'MAE': 'green',
    'MB+CONT': 'blue'
}

matplotlib.rcParams.update({'font.size': 36})
seeds = range(30)
paradigms = ['supervised', 'neurobooster', 'vicreg', 'simim', 'mae', 'bbworld']
folds = range(1)
labels_percentage_list = [100, 10, 1]

destination_folder_path = source_folder_path_all_experiments

results_mean_file = open(destination_folder_path / "results_mean.txt", "a", buffering=1)

fig, axs = plt.subplots(len(labels_percentage_list), 1, figsize=(22, 22))

# If only one plot, axs is an Axes object → convert to list for uniformity
if len(labels_percentage_list) == 1:
    axs = [axs]
else:
    axs = axs.flatten()

for row_idx, labels_percentage in enumerate(labels_percentage_list):
    metric = 'MAE'
    df = pd.read_csv(source_folder_path_all_experiments / f'results_collected_{metric}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv')

    paradigms_names = {'supervised': 'SL', 'neurobooster': 'MB', 'vicreg': 'VICReg', 'simim': 'SimMIM', 'baseline': 'baseline', 'mae': 'MAE', 'bbworld':'MB+CONT'}
    df = df.drop('Unnamed: 0', axis=1, errors='ignore')
    df.columns = [paradigms_names[col] for col in df.columns]
    df_results = df

    print('Plotting distributions')

    combined_data = pd.melt(df_results, var_name='Paradigm', value_name='Value')
    ax = sns.kdeplot(
        data=combined_data, x='Value', hue='Paradigm', fill=True,
        common_norm=True, bw_adjust=bw_adjust, cut=cut,
        ax=axs[row_idx], palette=paradigm_colors, legend=False
    )

    for paradigm in df_results.columns:
        color = paradigm_colors.get(paradigm, 'grey')
        sns.rugplot(data=df_results[paradigm], height=0.1, ax=axs[row_idx], color=color)

    axs[row_idx].set_xlabel('MAE (years)')
    axs[row_idx].set_ylabel('PDE')

    y_data = []
    for line in ax.get_lines():
        y_data.extend(line.get_ydata())
    if y_data:
        y_max = max(y_data)
        axs[row_idx].set_ylim(0, y_max * 1.1)

    axs[row_idx].spines['top'].set_visible(False)
    axs[row_idx].spines['right'].set_visible(False)
    axs[row_idx].spines['bottom'].set_visible(True)
    axs[row_idx].spines['left'].set_visible(True)

    print(f'\n{metric}, {labels_percentage}', file=results_mean_file)
    print(df_results.iloc[:, 1:].mean(), file=results_mean_file)

# Titles
row_titles = ['Labels percentage 100%', 'Labels percentage 1%']
for row_idx, row_title in enumerate(row_titles):
    fig.text(0.5, 0.92 - row_idx * 0.45, row_title, ha='center', va='bottom', fontsize=36)

# Legend
legend_handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in paradigm_colors.items()]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=len(paradigm_colors), title='Paradigm')

plt.subplots_adjust(hspace=0.6, top=0.90, bottom=0.1, left=0.08, right=0.95)

plt.savefig(destination_folder_path / f'new_center_title_combined_plots_MAE_mae_bbworld{bw_adjust}_{cut}_{dpi}dpi.png', dpi=dpi, bbox_inches='tight')
plt.close()

results_mean_file.close()
print("\n✅ Plotting and analysis complete!")
