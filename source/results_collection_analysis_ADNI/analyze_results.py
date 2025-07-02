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
paradigms = ['supervised', 'vicreg', 'bbworld', 'simim', 'neurobooster']
folds = range(1)
labels_percentage_list= [100, 10]
source_folder_path_all_experiments = Path('/home/andreaespis/diciotti/med-booster/EXPERIMENTS_MRI_augm_21_11')/'results_ADNI'
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

for metric in ['AUC', 'PR_AUC']:
    for labels_percentage in labels_percentage_list:
        df_results = pd.read_csv(source_folder_path_all_experiments /f'results_collected_{metric}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv')
        print(df_results)
        # after dropping not yet computed results:
        df_results = df_results[(df_results != 0).all(axis=1)]
        print(len(df_results))
        plot_distributions(df_results, destination_folder_path/f'metric_{metric}_labels_percentage_{labels_percentage}', metric)
        print(f'\n{metric}, {labels_percentage}', file = results_mean_file)

        print(df_results.iloc[:,1:].mean(), file = results_mean_file)




# comparative study:
# seeds = range(1)
# paradigms = ['supervised','vicreg', 'msn', 'pmsn']
# labels_percentage_list = [100,10]
# folds = range(5)
# source_folder_path_all_experiments = Path('/home/andreaespis/diciotti/self_unbalanced/EXPERIMENTS_2023_11_13_ARMAND_thesis_correct')/'results'
# destination_folder_path = source_folder_path_all_experiments


# import matplotlib.pyplot as plt
# import seaborn as sns

# # Create a scatter plot using Matplotlib

# paradigms_colors = {'supervised':'k','vicreg':'b', 'msn':'r', 'pmsn':'g'}
# paradigms_markers = {'supervised':'*','vicreg':'o', 'msn':'v', 'pmsn':'s'}
# for metric in ['auc', 'pr_auc']:
#     plt.figure(figsize=(10, 6))
#     df_results_100 = pd.read_csv(source_folder_path_all_experiments /f'results_collected_{metric.upper()}_seeds_{seeds}_folds_{folds}_labels_percentage_{100}_paradigms_{paradigms}.csv')
#     df_results_10 = pd.read_csv(source_folder_path_all_experiments /f'results_collected_{metric.upper()}_seeds_{seeds}_folds_{folds}_labels_percentage_{10}_paradigms_{paradigms}.csv')
#     for paradigm in ['supervised','vicreg', 'msn', 'pmsn']:
#         for cl0 in [10,30,50,70,90]:
#             cl1 = 100-cl0
#             per_class_results_100 = np.mean(np.array(df_results_100[paradigm][df_results_100.iloc[:,0].str.contains(f'cl0_{cl0}_cl1_{cl1}', case=False)]))
#             per_class_results_10 = np.mean(np.array(df_results_10[paradigm][df_results_100.iloc[:,0].str.contains(f'cl0_{cl0}_cl1_{cl1}', case=False)]))
#             plt.scatter(f'cl0_{cl0}_cl1_{cl1}', per_class_results_100, c = paradigms_colors[paradigm], s = 100, alpha = 1, marker = paradigms_markers[paradigm], edgecolors=paradigms_colors[paradigm])
#             plt.scatter(f'cl0_{cl0}_cl1_{cl1}', per_class_results_10, c = paradigms_colors[paradigm], s = 50, alpha = 0.5, marker = paradigms_markers[paradigm], edgecolors=paradigms_colors[paradigm])


#     plt.savefig(destination_folder_path/f'results_{metric}.png')