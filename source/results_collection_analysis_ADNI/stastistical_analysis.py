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
#python MAIN/source/results_collection_analysis/analyze_results.py



def stat_analysis(df, col1, col2, destination_path):

    print(f'{col1} vs {col2}', file = destination_path )

    #perform independent two sample t-test
    # An independent two sample t-test is used to determine if two population means are equal
    #print('independent two sample t-test:', ttest_ind(df[col1], df[col2]))

    # Paired Samples t-Test
    # A paired samples t-test is used to determine if two population means are equal
    # in which each observation in one sample can be paired with an observation in the other sample.
    print('One-tailed Paired Samples t-Test', ttest_rel(df[col1], df[col2],  alternative = 'greater'), file = destination_path) #alternative = 'less': the mean of the distribution underlying the first sample is less than the mean of the distribution underlying the second sample.

    #perform Welch's t-test
    # Welch’s t-test is similar to the independent two sample t-test, except it does not assume 
    # that the two populations that the samples came from have equal variance.
    #print("Welch's t-test", ttest_ind(df[col1], df[col2], equal_var=False), file = destination_path) # Welch's t-test: questo test non va bene perché non è per dati appaiati e si usa come alternativa al t-test quando si è molto lontani dall'omoschedasticità


    #wilcoxon(x, y=None, zero_method='wilcox', correction=False, alternative='two-sided', method='auto', *, axis=0, nan_policy='propagate', keepdims=False)[source]
    print("One-tailed Wilcoxon signed-rank", wilcoxon(df[col1], df[col2],  alternative = 'greater'), file = destination_path) # alternative = 'less. the distribution underlying d is stochastically less than a distribution symmetric about zero.

#One-tailed paired Samples t-Test: a noi interessa che neurobosster sia più performante degli altri (quindi che l'errore (RMSE o MAE) di neurobooster sia inferiore agli altri). 
#Farei quindi un test ad una sola coda (in python scipy dovrebbe esserci il parametro alternative per gestire questo)

#One-tailed Wilcoxon signed-rank test: il Wilcoxon test per dati appaitai si chiama Wilcoxon signed-rank test ed anche in questo caso lo devi fare ad una coda per il discorso fatto prima.

seeds = range(30)
paradigms = ['supervised', 'vicreg', 'bbworld', 'simim', 'neurobooster']
folds = range(1)
labels_percentage_list= [100, 10]
source_folder_path_all_experiments = Path('/home/andreaespis/diciotti/med-booster/EXPERIMENTS_MRI_augm_21_11')/'results_ADNI'
destination_folder_path = source_folder_path_all_experiments

destination_stats = open(destination_folder_path/'stats.txt' , "a", buffering=1)
for paradigms_to_compare in [['vicreg','neurobooster'],['neurobooster', 'bbworld'], ['neurobooster','vicreg'], ['neurobooster','supervised'], ['neurobooster','simim']]:
    for labels_percentage in labels_percentage_list:
        for metric in ['PR_AUC', 'AUC']:
            print(f'\n {metric}, {labels_percentage}:', file = destination_stats)
            df_results = pd.read_csv(source_folder_path_all_experiments /f'results_collected_{metric}_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv')
            print(df_results)
            df_results = df_results[(df_results != 0).all(axis=1)]
            print(df_results)
            stat_analysis(df_results, paradigms_to_compare[0], paradigms_to_compare[1],  destination_stats)


# Since the p-value is less than .05, we reject the null hypothesis of the t-test and 
# conclude that there is sufficient evidence to say that the two methods lead to 
# different mean exam scores.

# p-value <0.5 -> the two methods are different