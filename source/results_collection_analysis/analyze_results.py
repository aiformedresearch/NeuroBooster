################
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
#python MAIN/source/results_collection_analysis/analyze_results.py

seeds = range(30)
paradigms = ['vicreg', 'simim', 'neurobooster', 'supervised', 'bbworld']
folds = range(1)
labels_percentage_list= [100, 10]
source_folder_path_all_experiments = Path('/home/andreaespis/diciotti/med-booster/EXPERIMENTS_MRI_augm_21_11')/'results'
destination_folder_path = source_folder_path_all_experiments


def plot_distributions(df, save_path, metric):
    print('plotting distributions')
    plt.rcParams["figure.figsize"] = (10,2) 
    # Create the distribution plot
    sns.displot(df, kde=True, rug=True)
    # Add labels and title
    plt.xlabel(metric)
    plt.ylabel('# experiments')

    plt.savefig(f'{save_path}_distributions.png', dpi=300, bbox_inches='tight')

results_mean_file = open(destination_folder_path / "results_mean.txt", "a", buffering=1)

for metric in ['MAE', 'RMSE']:
    for labels_percentage in labels_percentage_list:
        df_results = pd.read_csv(source_folder_path_all_experiments /f'results_collected_{metric}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv')
        print(df_results)
        plot_distributions(df_results, destination_folder_path/f'metric_{metric}_labels_percentage_{labels_percentage}', metric)
        print(f'\n{metric}, {labels_percentage}', file = results_mean_file)
        print(df_results.iloc[:,1:].mean(), file = results_mean_file)

# for metric in ['RMSE', 'MAE']:
#     for labels_percentage in labels_percentage_list:
#         df_results = pd.read_csv(source_folder_path_all_experiments /f'results_collected_{metric}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv')
#         print(df_results)
#         plot_distributions(df_results, destination_folder_path/f'metric_{metric}_labels_percentage_{labels_percentage}')
