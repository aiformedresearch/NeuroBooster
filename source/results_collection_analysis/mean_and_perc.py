from scipy.stats import ttest_rel
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

# === Configuration ===
dataset = 'AGE'  # Change to 'AGE' to switch dataset
seeds = range(30)
folds = range(1)
labels_percentage_list = [100, 10, 1]
paradigms = ['supervised', 'neurobooster', 'simclr', 'vicreg', 'mae', 'simim']

# === Dataset-specific settings ===
if dataset == 'ADNI':
    metrics = ['AUC', 'PR_AUC']
    source_folder_path_all_experiments = Path('/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS/results_definitive_ADNI')
    plot_title = 'ADNI, backbone frozen during fine-tuning'
elif dataset == 'AGE':
    metrics = ['MAE', 'RMSE']
    source_folder_path_all_experiments = Path('/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS/results_definitive_AGE')
    plot_title = 'AGE, backbone frozen during fine-tuning'
else:
    raise ValueError("Invalid dataset. Must be 'ADNI' or 'AGE'.")

destination_folder_path = source_folder_path_all_experiments
alpha_corrected = 0.05

def get_analysis_result(df_path, metric, alpha_corrected):
    df = pd.read_csv(df_path)
    df = df.drop('Unnamed: 0', axis=1, errors='ignore')
    paradigms_names = {'supervised': 'SL', 'neurobooster': 'NB', 'vicreg': 'VICReg', 'simim': 'SimMIM', 'baseline': 'baseline', 'mae': 'MAE', 'simclr': 'SimCLR'}
    df.columns = [paradigms_names.get(col, col) for col in df.columns]
    confidence_intervals_df = pd.DataFrame(index=df.columns, columns=[f'{metric}', f'({metric}) p-value < {alpha_corrected}'])

    list_mean, list_margin_of_error, list_2_5_percentile, list_97_5_percentile, list_std = [], [], [], [], []

    for column in df.columns:
        data = df[column]
        list_2_5_percentile.append(np.percentile(data, 2.5))
        list_97_5_percentile.append(np.percentile(data, 97.5))
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        list_std.append(std_dev)
        n = len(data)
        z_score = stats.norm.ppf(0.975) if n >= 30 else stats.t.ppf(0.975, df=n-1)
        margin_of_error = z_score * (std_dev / np.sqrt(n))

        list_mean.append(round(mean, 2))
        list_margin_of_error.append(round(margin_of_error, 2))
        output_string = f"{mean:.3f} ± {margin_of_error:.3f}"

        # Determine whether lower or higher is better
        alternative = 'less' if metric in ['MAE', 'RMSE'] else 'greater'
        significant_columns = [col for col in df.columns if col != column and stats.ttest_rel(df[column], df[col], alternative=alternative).pvalue < alpha_corrected]

        confidence_intervals_df.loc[column, f'{metric}'] = output_string
        confidence_intervals_df.loc[column, f'({metric}) p-value < {alpha_corrected}'] = ', '.join(significant_columns)

    paired_values = list(zip(list_mean, list_margin_of_error))
    reverse = False if metric in ['MAE', 'RMSE'] else True
    sorted_indices = sorted(range(len(paired_values)), key=lambda k: (paired_values[k][0], -paired_values[k][1]), reverse=reverse)
    greatest_mean_indices = sorted_indices[:2]
    same_score = (list_mean[greatest_mean_indices[0]] == list_mean[greatest_mean_indices[1]] and
                  list_margin_of_error[greatest_mean_indices[0]] == list_margin_of_error[greatest_mean_indices[1]])

    confidence_intervals_df['2.5 percentile'] = list_2_5_percentile
    confidence_intervals_df['97.5 percentile'] = list_97_5_percentile
    confidence_intervals_df['std'] = list_std
    return confidence_intervals_df, greatest_mean_indices, same_score


for labels_percentage in labels_percentage_list:
    confidence_tables = []
    indices_highlight = []
    same_scores = []

    for metric in metrics:
        df_path = source_folder_path_all_experiments / f'results_collected_{metric}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv'
        metric_df, top_indices, same = get_analysis_result(df_path, metric, alpha_corrected)

        # === PRINT METRIC TABLE TO CONSOLE ===
        print(f"\nDataset: {dataset} - Metric: {metric} - Labels Percentage: {labels_percentage}")
        print(metric_df)

        confidence_tables.append(metric_df)
        indices_highlight.append((metric, top_indices))
        same_scores.append(same)

    confidence_intervals_df = pd.concat(confidence_tables, axis=1)

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.axis('off')
    table = plt.table(cellText=confidence_intervals_df.values,
                      colLabels=confidence_intervals_df.columns,
                      rowLabels=confidence_intervals_df.index,
                      loc='center',
                      cellLoc='center',
                      colColours=['#f5f5f5'] * confidence_intervals_df.shape[1],
                      bbox=[0, 0, 1, 1])

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold')
        elif j == -1:
            cell.set_text_props(weight='bold')
        cell.set_text_props(ha="center")

    for (metric, top_indices), same in zip(indices_highlight, same_scores):
        col_start = confidence_intervals_df.columns.get_loc(f'{metric}')
        first = False
        for row_idx in top_indices:
            table[(row_idx + 1, col_start)].set_text_props(weight='bold')
            if not first and not same:
                first = True
                text = table[(row_idx + 1, col_start)].get_text()
                text.set_text('{:s}'.format('\u0332'.join(text.get_text())))

    table.auto_set_column_width(range(confidence_intervals_df.shape[1]))
    table.set_fontsize(12)
    plt.title(f'{plot_title}, labels {labels_percentage}%', fontsize=16, y=1.1)
    plt.savefig(destination_folder_path / f'combined_info_{dataset}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.png', bbox_inches='tight')

    latex_table = confidence_intervals_df.to_latex()
    with open(destination_folder_path / f'latex_combined_info_{dataset}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.tex', 'w') as f:
        f.write(latex_table)
