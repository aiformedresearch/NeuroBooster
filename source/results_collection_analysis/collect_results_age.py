from pathlib import Path
import os
import pandas as pd
#python MAIN/source/results_collection_analysis/collect_results.py

seeds = range(30)
paradigms =  ['supervised', 'neurobooster', 'vicreg', 'simim', 'mae', 'bbworld']
folds = range(1)
labels_percentage_list= [100,10, 1]
source_folder_path_all_experiments = Path('/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS/')
destination_folder_path = source_folder_path_all_experiments/ f'results_AGE/' 
os.makedirs(destination_folder_path, exist_ok=True)

indexes = []
for seed in seeds:
  for fold in folds:
      indexes.append(f'seed_{seed}_fold_{fold}')
data = {paradigm:[0.0]*len(indexes) for paradigm in paradigms}


results_collected_MAE = {labels_percentage: pd.DataFrame(data, index = indexes) for labels_percentage in labels_percentage_list}

for seed in seeds:
  for dataset_name in ['AGE']:
    for paradigm in paradigms: 
      print(paradigm)
      for labels_percentage in labels_percentage_list: 
          for k_fold in folds:
            folder_name = f'seed{seed}' 
            source_folder_path = source_folder_path_all_experiments / folder_name
            file_path_finetuning = source_folder_path / dataset_name / paradigm / f'labels_percentage_{labels_percentage}'/ f'fold_{k_fold}' / 'finetuning_metrics'/ 'finetuning_df_metrics.csv'
            file_path_validation = source_folder_path / dataset_name / paradigm / f'labels_percentage_{labels_percentage}'/ f'fold_{k_fold}' / 'val_metrics'/ 'val_df_metrics.csv'
            df_finetune = pd.read_csv(file_path_finetuning)
            df_val = pd.read_csv(file_path_validation)
            idx_min_finetune_loss = df_finetune[['fine_tuning_loss']].idxmin() 
            val_best_MAE = df_val['val_MAE_loss'][idx_min_finetune_loss.iloc[0]] 
            print(idx_min_finetune_loss.iloc[0])
            results_collected_MAE[labels_percentage][paradigm][f'seed_{seed}_fold_{k_fold}'] = val_best_MAE

for labels_percentage in labels_percentage_list:        
    results_collected_MAE[labels_percentage].to_csv(destination_folder_path/ f'results_collected_MAE_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv', index=True)
    print(f'\nMAE labels percentage {labels_percentage}:\n',results_collected_MAE[labels_percentage])



# for seed in seeds:
#   for dataset_name in ['ADNI']:
#     for paradigm in paradigms: 
#       print(paradigm)
#       for labels_percentage in labels_percentage_list: 
#           for k_fold in folds:
#             folder_name = f'EXPS/seed{seed}' 
#             source_folder_path = source_folder_path_all_experiments / folder_name
#             file_path_finetuning = source_folder_path / dataset_name / paradigm / f'labels_percentage_{labels_percentage}'/ f'fold_{k_fold}' / 'finetuning_metrics'/ 'finetuning_df_metrics.csv'
#             file_path_validation = source_folder_path / dataset_name / paradigm / f'labels_percentage_{labels_percentage}'/ f'fold_{k_fold}' / 'val_metrics'/ 'val_df_metrics.csv'
#             df_finetune = pd.read_csv(file_path_finetuning)
#             df_val = pd.read_csv(file_path_validation)
#             idx_min_finetune_loss = df_finetune[['fine_tuning_loss']].idxmin() 
#             val_best_AUC = df_val['auc'][idx_min_finetune_loss.iloc[0]] 
#             val_best_PR_AUC = df_val['pr_auc'].iloc[idx_min_finetune_loss.iloc[0]] 
#             print(idx_min_finetune_loss.iloc[0])
#             print(val_best_AUC)
#             results_collected_AUC[labels_percentage][paradigm][f'seed_{seed}_fold_{k_fold}'] = val_best_AUC
#             results_collected_PR_AUC[labels_percentage][paradigm][f'seed_{seed}_fold_{k_fold}'] = val_best_PR_AUC

# for labels_percentage in labels_percentage_list:        
#     results_collected_AUC[labels_percentage].to_csv(destination_folder_path/ f'results_collected_AUC_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv', index=True)
#     print(f'\nAUC labels percentage {labels_percentage}:\n',results_collected_AUC[labels_percentage])
#     results_collected_PR_AUC[labels_percentage].to_csv(destination_folder_path/ f'results_collected_PR_AUC_seeds_{seeds}_folds_{folds}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv', index=True)
#     print(f'\nPR_AUC labels percentage {labels_percentage}:\n',results_collected_PR_AUC[labels_percentage])

