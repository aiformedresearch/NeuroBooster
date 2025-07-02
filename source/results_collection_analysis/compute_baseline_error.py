import sys
 
# adding Folder_2/subfolder to the system path
sys.path.insert(0, '/home/andreaespis/diciotti/med-booster/MAIN/source/utils')
sys.path.insert(0, '/home/andreaespis/diciotti/med-booster/MAIN/source')
import general_utils
import dataset_utils

import numpy as np
import json
from pathlib import Path
import pandas as pd

folds = range(4)
paradigms = ['neurobooster','bbworld']
tabular_dir='/home/andreaespis/diciotti/data/AGE_prediction/05_10_2023/NF_Andrea_part1and2.csv'
destination_folder_path = Path('/home/andreaespis/diciotti/med-booster/EXPERIMENTS_2023_10_05_debug_same_as_bbworld_2')/'results'
seeds = [0]

class arguments:
    def __init__(self):
        self.exp_dir = Path('/home/andreaespis/diciotti/med-booster/EXPERIMENTS_2023_10_05_debug_same_as_bbworld_2')
        self.paradigm = 'supervised'
        self.cross_val_folds = len(folds)
        self.tabular_dir = tabular_dir
        self.labels_percentage = 100
        self.task = 'regression'
        self.quick_check_reproducibility_file = False


args = arguments()
results_collected = pd.read_csv(destination_folder_path/ f'results_collected_seeds_{seeds}_folds_{folds}_paradigms_{paradigms}.csv',index_col=[0])
results_collected['baseline']=[0.0]*len(results_collected.iloc[:,0])


print(results_collected)

for seed in seeds:
    general_utils.set_reproducibility(seed)
    indexes_train_folds, indexes_val_folds, targets_train_folds, targets_val_folds, all_targets, tabular_scaler_folds, args.num_classesr = dataset_utils.get_samples_indexes_for_each_fold(args) 

    for fold in folds:
        print(f'############# fold {fold} ##############')

        for label_percentage in [100]:


            conc_targets_train = targets_train_folds[fold]
            conc_targets_val = targets_val_folds[fold]
            scaler = tabular_scaler_folds[fold]

            conc_targets_train = np.array(conc_targets_train[:int(len(conc_targets_train)*label_percentage/100)])

            conc_targets_train = scaler.inverse_transform(conc_targets_train.reshape(-1, 1)) 
            conc_targets_train = list(conc_targets_train[:,0])

            mean_train = np.mean(conc_targets_train)
            conc_targets_val = list(scaler.inverse_transform(np.array(conc_targets_val).reshape(-1, 1)))
            baseline_error = np.mean(np.array([(cv - mean_train)**2 for cv in conc_targets_val]))

            print(baseline_error)
            results_collected.loc[f'seed_{seed}_fold_{fold}','baseline'] = baseline_error**0.5

paradigms = paradigms + ['baseline']
results_collected.to_csv(destination_folder_path/ f'results_collected_seeds_{seeds}_folds_{folds}_paradigms_{paradigms}.csv')
print(results_collected)