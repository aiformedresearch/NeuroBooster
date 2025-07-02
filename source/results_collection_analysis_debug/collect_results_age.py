from pathlib import Path
import os
import pandas as pd
import numpy as np

seeds = range(5)
paradigms = ['medbooster', 'supervised']
metric = 'MAE'  # or 'MAE'
folds = range(1)
labels_percentage_list = [100, 1]

source_folder_path_all_experiments = Path('/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_16_LARS_long_no_augm')
destination_folder_path = source_folder_path_all_experiments / 'results_AGE'
os.makedirs(destination_folder_path, exist_ok=True)

# Prepare result structure
indexes = [f'seed_{seed}_fold_{fold}' for seed in seeds for fold in folds]
data = {paradigm: [0.0] * len(indexes) for paradigm in paradigms}
results_collected_metric = {
    labels_percentage: pd.DataFrame(data, index=indexes)
    for labels_percentage in labels_percentage_list
}

# Main loop
for seed in seeds:
    for dataset_name in ['AGE']:
        for paradigm in paradigms:
            print(paradigm)
            for labels_percentage in labels_percentage_list:
                for k_fold in folds:
                    folder_name = f'seed{seed}'
                    source_folder_path = source_folder_path_all_experiments / folder_name
                    base_path = source_folder_path / dataset_name / paradigm / f'labels_percentage_{labels_percentage}' / f'fold_{k_fold}'

                    file_path_finetuning = base_path / 'finetuning_metrics' / 'finetuning_df_metrics.csv'
                    file_path_validation = base_path / 'val_metrics' / 'val_df_metrics.csv'

                    try:
                        df_finetune = pd.read_csv(file_path_finetuning)
                        df_val = pd.read_csv(file_path_validation)

                        # Get best epoch from fine-tuning loss
                        min_row = df_finetune.loc[df_finetune['fine_tuning_loss'].idxmin()]
                        epoch_number = int(min_row['epoch'])
                        print(f"Best epoch based on fine_tuning_loss: {epoch_number}")

                        # Match epoch in validation metrics
                        val_row = df_val[df_val['epoch'] == epoch_number]
                        if val_row.empty:
                            print(f"⚠️ Epoch {epoch_number} not found in df_val, skipping...")
                            continue
                        val_row = val_row.iloc[0]

                        if metric == 'MAE':
                            val_best_metric = val_row[f'val_{metric}_loss']

                        elif metric == 'RMSE':
                            # Use the (typoed) folder name
                            output_folder = base_path / 'valdiation_outputs_and_targets'
                            preds_path = output_folder / f'y_predicted_validation_epoch{epoch_number}.npy'
                            targets_path = output_folder / f'y_target_validation_epoch{epoch_number}.npy'

                            if not preds_path.exists() or not targets_path.exists():
                                print(f"⚠️ Missing files for epoch {epoch_number}, skipping...\n{preds_path}\n{targets_path}")
                                continue

                            y_pred = np.load(preds_path).astype(np.float64)
                            y_true = np.load(targets_path).astype(np.float64)

                            # Check arrays for invalid values
                            if not np.all(np.isfinite(y_pred)):
                                print(f"⚠️ y_pred contains NaN or inf at epoch {epoch_number}, skipping...")
                                continue
                            if not np.all(np.isfinite(y_true)):
                                print(f"⚠️ y_true contains NaN or inf at epoch {epoch_number}, skipping...")
                                continue

                            # Compute scaling factor
                            val_loss = val_row['val_loss']
                            val_rescaled_loss = val_row['val_rescaled_loss']
                            print(f"  - Epoch {epoch_number} — val_loss: {val_loss}, val_rescaled_loss: {val_rescaled_loss}")

                            if pd.isna(val_loss) or pd.isna(val_rescaled_loss) or val_loss == 0:
                                print(f"⚠️ Invalid val_loss or rescaled loss, skipping...")
                                continue

                            scaling_factor = val_rescaled_loss / val_loss
                            if not np.isfinite(scaling_factor):
                                print(f"⚠️ Invalid scaling factor: {scaling_factor}, skipping...")
                                continue

                            print(f"  - Scaling factor: {scaling_factor}")
                            print(f"  - y_pred range: min={y_pred.min()}, max={y_pred.max()}")
                            print(f"  - y_true range: min={y_true.min()}, max={y_true.max()}")

                            y_pred_rescaled = y_pred * scaling_factor
                            y_true_rescaled = y_true * scaling_factor

                            print(f"  - y_pred_rescaled range: min={y_pred_rescaled.min()}, max={y_pred_rescaled.max()}")
                            print(f"  - y_true_rescaled range: min={y_true_rescaled.min()}, max={y_true_rescaled.max()}")

                            if not np.all(np.isfinite(y_pred_rescaled)) or not np.all(np.isfinite(y_true_rescaled)):
                                print(f"⚠️ Rescaled predictions/targets contain NaN or inf, skipping...")
                                continue

                            # Optional: clip extreme values
                            # y_pred_rescaled = np.clip(y_pred_rescaled, -1e6, 1e6)
                            # y_true_rescaled = np.clip(y_true_rescaled, -1e6, 1e6)

                            val_best_metric = np.sqrt(np.mean((y_pred_rescaled - y_true_rescaled) ** 2))

                            if not np.isfinite(val_best_metric):
                                print(f"⚠️ Computed RMSE is not finite, skipping...")
                                continue

                        else:
                            raise ValueError(f"Unsupported metric: {metric}")

                        # Save result (safe assignment)
                        results_collected_metric[labels_percentage].loc[f'seed_{seed}_fold_{k_fold}', paradigm] = val_best_metric

                    except Exception as e:
                        print(f"⚠️ Error in seed {seed}, paradigm {paradigm}, labels {labels_percentage}, fold {k_fold}: {e}")
                        continue

# Save results
for labels_percentage in labels_percentage_list:
    output_file = destination_folder_path / (
        f'results_collected_{metric}_seeds_{list(seeds)}_folds_{list(folds)}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv'
    )
    results_collected_metric[labels_percentage].to_csv(output_file, index=True)
    print(f'\n{metric} labels percentage {labels_percentage}:\n', results_collected_metric[labels_percentage])

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

