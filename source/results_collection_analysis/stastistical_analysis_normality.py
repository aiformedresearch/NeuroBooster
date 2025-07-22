from scipy.stats import ttest_rel, ttest_ind, wilcoxon, shapiro
import numpy as np
import pandas as pd
from pathlib import Path

def stat_analysis(df, col1, col2, destination_path, metric):

    print(f'{col1} vs {col2}', file=destination_path)

    # Determine the correct alternative hypothesis
    if metric in ['MAE', 'RMSE']:
        alt = 'less'  # lower is better
    else:
        alt = 'greater'  # higher is better

    # Normality tests
    p1 = shapiro(df[col1])[1]
    p2 = shapiro(df[col2])[1]

    if p1 > 0.05 and p2 > 0.05:
        print("Both metrics normally distributed", file=destination_path)
    elif p1 > 0.05 or p2 > 0.05:
        print("Only one metric normally distributed", file=destination_path)
    else:
        print("None of the metrics normally distributed", file=destination_path)

    print(f'Shapiro p-values: {col1} = {p1:.4f}, {col2} = {p2:.4f}', file=destination_path)

    # Paired Samples t-Test
    ttest_result = ttest_rel(df[col1], df[col2], alternative=alt)
    print(f'One-tailed Paired Samples t-Test (alternative="{alt}")\n{ttest_result}\n', file=destination_path)

    # Wilcoxon Signed-Rank Test
    try:
        wilcoxon_result = wilcoxon(df[col1], df[col2], alternative=alt)
        print(f'One-tailed Wilcoxon signed-rank test (alternative="{alt}")\n{wilcoxon_result}\n', file=destination_path)
    except ValueError as e:
        print(f'Wilcoxon test failed: {e}', file=destination_path)

# === Setup ===
seeds = range(30)
paradigms = ['supervised', 'neurobooster', 'simclr', 'vicreg', 'mae', 'simim']
folds = range(1)
labels_percentage_list = [100, 10, 1]
dataset = 'ADNI'
source_folder_path_all_experiments = Path('/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS') / f'results_definitive_{dataset}'
destination_folder_path = source_folder_path_all_experiments

if dataset == 'AGE':
    metrics = ['RMSE', 'MAE']
else:
    metrics = ['AUC', 'PR_AUC']

destination_stats = open(destination_folder_path / 'stats_normality.txt', "a", buffering=1)

# === Loop through comparisons ===
for paradigms_to_compare in [['neurobooster', 'supervised'], ['neurobooster', 'simclr'], ['neurobooster', 'vicreg'], ['neurobooster', 'mae'], ['neurobooster', 'simim']]:
    for labels_percentage in labels_percentage_list:
        for metric in metrics:
            print(f'\n{metric}, labels {labels_percentage}%:', file=destination_stats)
            df_results = pd.read_csv(
                source_folder_path_all_experiments /
                f'results_collected_{metric}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv'
            )
            stat_analysis(df_results, paradigms_to_compare[0], paradigms_to_compare[1], destination_stats, metric)
