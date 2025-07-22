from scipy.stats import ttest_rel,  ttest_ind, wilcoxon


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
from matplotlib.colors import ListedColormap
#python MAIN/source/results_collection_analysis/analyze_results.p

import pandas as pd
from scipy import stats


seeds = range(30)
paradigms = ['supervised', 'neurobooster', 'simclr', 'vicreg', 'mae', 'simim']
folds = range(1)
labels_percentage_list= [100, 10, 1]
source_folder_path_all_experiments = Path('/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS')/'results_definitive_ADNI'
destination_folder_path = source_folder_path_all_experiments
alpha_corrected = 0.05

def get_analysis_result(df_path, metric, alpha_corrected):
        df = pd.read_csv(df_path)
        df = df.drop('Unnamed: 0', axis=1, errors='ignore')
        paradigms_names = {'supervised': 'SL', 'neurobooster': 'NB', 'vicreg': 'VICReg', 'simim': 'SimMIM', 'baseline': 'baseline', 'mae': 'MAE', 'simclr': 'SimCLR'}#, 'baseline':'baseline'}
        df = df.drop('Unnamed: 0', axis=1, errors='ignore')
        df.columns = [paradigms_names[col] for col in df.columns]
        confidence_intervals_df = pd.DataFrame(index=df.columns, columns=[f'{metric}', f'({metric}) p-value < {alpha_corrected}'])
       
        # Loop through each column of the original DataFrame
        list_mean =[]
        list_margin_of_error = []
        list_2_5_percentile = []
        list_97_5_percentile = []
        list_std = []
        for column in df.columns:
            # Extract the values from the column
            data = df[column]
            list_2_5_percentile.append(np.percentile(data, 2.5))  # return 2.5th percentile
            list_97_5_percentile.append(np.percentile(data, 97.5))  # return 2.5th percentile
            # Calculate sample mean and standard deviation
            mean = np.mean(data)

            std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
            list_std.append(std_dev)
            # Sample size
            n = len(data)
            
            # Confidence level
            confidence_level = 0.95
            
            # Calculate margin of error
            if n >= 30:
                z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            else:
                z_score = stats.t.ppf(1 - (1 - confidence_level) / 2, df=n - 1)
            
            margin_of_error = z_score * (std_dev / np.sqrt(n))

            list_mean.append(round(mean,2))
            list_margin_of_error.append(round(margin_of_error,2))
            
            # Format the output string
            output_string = f"{mean:.3f} ± {margin_of_error:.3f}"
            
            # Perform t-test for each pair of columns
            significant_columns = [col for col in df.columns if col != column]
            significant_columns = [col for col in significant_columns if stats.ttest_rel(df[column], df[col],  alternative = 'greater').pvalue < 0.05]
            # Store the results in the new DataFrame
            confidence_intervals_df.loc[column, f'{metric}'] = output_string
            confidence_intervals_df.loc[column, f'({metric}) p-value < {alpha_corrected}'] = ', '.join(significant_columns)#str(significant_columns).strip('[').strip(']').strip("'")

        # Pair elements from both lists using zip
        paired_values = list(zip(list_mean, list_margin_of_error))

        # Find the indices based on the maximum values in the first list
        # If values in list1 are the same, use the minimum value in list2 to determine the order
        sorted_indices = sorted(range(len(paired_values)), key=lambda k: (paired_values[k][0], -paired_values[k][1]), reverse=True)

        # Get the indices of the two maximum values
        greatest_mean_indices = sorted_indices[:2]
        same_score = False
        if (list_mean[greatest_mean_indices[0]]==list_mean[greatest_mean_indices[1]]) and (list_margin_of_error[greatest_mean_indices[0]]==list_margin_of_error[greatest_mean_indices[1]]):
             same_score = True
             
        confidence_intervals_df = confidence_intervals_df.drop('mean', axis=1, errors='ignore')
        confidence_intervals_df['2.5 percentile'] = list_2_5_percentile
        confidence_intervals_df['97.5 percentile'] = list_97_5_percentile
        confidence_intervals_df['std'] = list_std
        return confidence_intervals_df, greatest_mean_indices, same_score


for labels_percentage in labels_percentage_list:
        auc_df_path = source_folder_path_all_experiments /f'results_collected_AUC_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv'
        pr_auc_df_path = source_folder_path_all_experiments /f'results_collected_PR_AUC_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv'

        auc_df, greatest_mean_indices_auc, same_score_auc = get_analysis_result(auc_df_path, 'AUC', alpha_corrected)
        pr_auc_df, greatest_mean_indices_pr_auc, same_score_pr_auc= get_analysis_result(pr_auc_df_path, 'PR_AUC', alpha_corrected)        
        print('\n\n')
        print(f'labels percentage: {labels_percentage}')
        print(pr_auc_df)

        confidence_intervals_df = pd.concat([auc_df, pr_auc_df], axis=1)
        # Save the DataFrame as an image
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        table = plt.table(cellText=confidence_intervals_df.values,
                        colLabels=confidence_intervals_df.columns,
                        rowLabels=confidence_intervals_df.index,
                        loc='center',
                        cellLoc='center',
                        colColours=['#f5f5f5']*confidence_intervals_df.shape[1],
                        bbox=[0, 0, 1, 1])

        # Rotate column labels for better readability
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
            elif j == -1:
                cell.set_text_props(weight='bold')  # Bold row labels
            
            cell.set_text_props(ha="center")


        first_row = False
        first_column = 0
        for row_idx in greatest_mean_indices_auc:
            table[(row_idx+1,first_column)].set_text_props(weight='bold')
            if not(first_row):
                if not(same_score_auc): first_row=True
                text = table[(row_idx+1,first_column)].get_text()
                text.set_text('{:s}'.format('\u0332'.join( text.get_text())))
            
        first_row = False
        first_column = 2
        for row_idx in greatest_mean_indices_pr_auc:
            table[(row_idx+1,first_column)].set_text_props(weight='bold')
            if not(first_row):
                if not(same_score_pr_auc): first_row=True
                text = table[(row_idx+1,first_column)].get_text()
                text.set_text('{:s}'.format('\u0332'.join( text.get_text())))

        # Adjust column widths
        table.auto_set_column_width([0, 1,2,3])
        table.set_fontsize(12)
        #plt.subplots_adjust(bottom=0.2, top=0.7, left=0.3, right=0.6)  # Adjust bottom to reduce unused space
        plt.title(f'ADNI, backbone frozen during fine-tuning, labels {labels_percentage}%', fontsize=16, y=1.1)
        plt.savefig(destination_folder_path/f'combined_info_AUC_and_PR_AUC_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.png', bbox_inches='tight')

        # Export the DataFrame to LaTeX
        latex_table = confidence_intervals_df.to_latex()

        # Print or save the LaTeX code to a file
        # Optionally, save to a file
        with open(destination_folder_path/f'latex_combined_info_AUC_and_PR_AUC_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.tex', 'w') as f:
            f.write(latex_table)