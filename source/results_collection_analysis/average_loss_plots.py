from pathlib import Path
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#python MAIN/source/results_collection_analysis/collect_results.py

seeds = [0,1]
paradigms = ['supervised','bbworld']
folds = range(0)
source_folder_path_all_experiments = Path('/home/andreaespis/diciotti/med-booster/EXPERIMENTS_2023_10_05_debug_same_as_bbworld_2')
destination_folder_path = source_folder_path_all_experiments/ f'results/' 
os.makedirs(destination_folder_path, exist_ok=True)

def save_plot_loss(loss, title, path):
   plt.title(title)
   print(loss.shape)
   plt.plot(range(len(loss)), loss, color="blue")
   plt.savefig(path)
   plt.close()

   

indexes = []
for seed in seeds:
  for fold in folds:
    indexes.append(f'seed_{seed}_fold_{fold}')
data = {paradigm:[0.0]*len(indexes) for paradigm in paradigms}
results_collected = pd.DataFrame(data, index = indexes)



for paradigm in paradigms: #['supervised', 'self_supervised', 'vicreg']:
    train_loss = []
    fine_tune_loss = []
    val_loss = []
    for seed in seeds:
        for dataset_name in ['AGE']:      
            for labels_percentage in [100, 10]: 
                for k_fold in folds:
                    print(paradigm, seed, k_fold)
                    folder_name = f'EXPS/seed{seed}' 
                    source_folder_path = source_folder_path_all_experiments / folder_name
                    file_path_pretraining = source_folder_path / dataset_name / paradigm / f'labels_percentage_{labels_percentage}'/ f'fold_{k_fold}' / 'pretraining_metrics'/ 'train_df_metrics.csv'
                    file_path_finetuning = source_folder_path / dataset_name / paradigm / f'labels_percentage_{labels_percentage}'/ f'fold_{k_fold}' / 'finetuning_metrics'/ 'finetuning_df_metrics.csv'
                    file_path_validation = source_folder_path / dataset_name / paradigm / f'labels_percentage_{labels_percentage}'/ f'fold_{k_fold}' / 'val_metrics'/ 'val_df_metrics.csv'
                    df_train = pd.read_csv(file_path_pretraining)
                    df_finetune = pd.read_csv(file_path_finetuning)
                    df_val = pd.read_csv(file_path_validation)
                    train_loss.append(np.array(df_train['loss']))
                    fine_tune_loss.append(np.array(df_finetune[['fine_tuning_loss']]))
                    val_loss.append(np.array(df_val['val_loss']))

    lengths = [len(loss) for loss in train_loss]
    train_loss =np.mean(np.array([loss[:min(lengths)] for loss in train_loss]), axis = 0)
    lengths = [len(loss) for loss in fine_tune_loss]
    fine_tune_loss = np.mean(np.array([loss[:min(lengths)] for loss in fine_tune_loss]), axis = 0)
    lengths = [len(loss) for loss in val_loss]
    val_loss = np.mean(np.array([loss[:min(lengths)] for loss in val_loss]), axis = 0)


    save_plot_loss(train_loss, f'pretrain_loss_{paradigm}', destination_folder_path/ f'avg_pretrain_loss_{seeds}_folds_{folds}_{paradigm}.png')        
    save_plot_loss(fine_tune_loss, f'finetuning_loss_{paradigm}', destination_folder_path/ f'avg_fine_tune_loss_{seeds}_folds_{folds}_{paradigm}.png')        
    save_plot_loss(val_loss, f'val_loss_{paradigm}', destination_folder_path/ f'avg_val_loss_{seeds}_folds_{folds}_{paradigm}.png')        





        
        

