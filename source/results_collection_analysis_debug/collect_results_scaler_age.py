from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import re
import joblib

# === Settings ===
seeds = range(1)
paradigms = ['supervised']
folds = range(1)
labels_percentage_list = [100, 1]
dataset_name = 'AGE'

base_path = Path('/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS')
destination_path = base_path / 'results_AGE_corrected'

# === Init ===
indexes = [f'seed_{seed}_fold_{fold}' for seed in seeds for fold in folds]
def init_metric_dict():
    return {lp: pd.DataFrame({paradigm: [0.0] * len(indexes) for paradigm in paradigms}, index=indexes)
            for lp in labels_percentage_list}

# MAE & RMSE dicts
val_mae_file = init_metric_dict()
val_rmse_file = init_metric_dict()

mae_from_log = init_metric_dict()
rmse_from_log = init_metric_dict()

mae_from_scaler_file = init_metric_dict()
rmse_from_scaler_file = init_metric_dict()

# === Scaler from log file ===
def parse_scaler_from_log(log_path):
    try:
        with open(log_path, 'r') as f:
            text = f.read()
        mean_match = re.search(r'before scaling train set age mean ([\d.]+)', text)
        std_match = re.search(r'std ([\d.]+)', text)
        if mean_match and std_match:
            mean = float(mean_match.group(1))
            std = float(std_match.group(1))
            scaler = StandardScaler()
            scaler.mean_ = np.array([mean])
            scaler.scale_ = np.array([std])
            return scaler
    except Exception as e:
        print(f"⚠️ Failed to parse scaler from {log_path}: {e}")
    return None

# === Main loop ===
for seed in seeds:
    for paradigm in paradigms:
        for labels_percentage in labels_percentage_list:
            for k_fold in folds:
                idx = f'seed_{seed}_fold_{k_fold}'
                exp_folder = base_path / f'seed{seed}' / dataset_name / paradigm / f'labels_percentage_{labels_percentage}'
                fold_folder = exp_folder / f'fold_{k_fold}'

                try:
                    # Paths
                    finetune_metrics = fold_folder / 'finetuning_metrics' / 'finetuning_df_metrics.csv'
                    val_metrics = fold_folder / 'val_metrics' / 'val_df_metrics.csv'
                    log_path = exp_folder / 'finetuning_output.log'
                    scaler_file_path = fold_folder / 'tabular_scaler.pkl'

                    # Load metrics
                    df_finetune = pd.read_csv(finetune_metrics)
                    df_val = pd.read_csv(val_metrics)
                    idx_min_loss = df_finetune['fine_tuning_loss'].idxmin()
                    best_epoch = int(df_val.loc[idx_min_loss, 'epoch'])
                    val_mae = df_val.loc[idx_min_loss, 'val_MAE_loss']
                    val_rmse = df_val.loc[idx_min_loss, 'val_RMSE_loss'] if 'val_RMSE_loss' in df_val.columns else np.nan

                    val_mae_file[labels_percentage].loc[idx, paradigm] = val_mae
                    val_rmse_file[labels_percentage].loc[idx, paradigm] = val_rmse

                    # Load predictions
                    y_pred = np.load(fold_folder / f'valdiation_outputs_and_targets/y_predicted_validation_epoch{best_epoch}.npy')
                    y_true = np.load(fold_folder / f'valdiation_outputs_and_targets/y_target_validation_epoch{best_epoch}.npy')

                    # === With log-based scaler
                    scaler_log = parse_scaler_from_log(log_path)
                    if scaler_log is None:
                        raise ValueError("Scaler from log failed.")
                    y_pred_log = scaler_log.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    y_true_log = scaler_log.inverse_transform(y_true.reshape(-1, 1)).flatten()
                    mae_log = mean_absolute_error(y_true_log, y_pred_log)
                    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
                    mae_from_log[labels_percentage].loc[idx, paradigm] = mae_log
                    rmse_from_log[labels_percentage].loc[idx, paradigm] = rmse_log

                    # === With scaler.pkl
                    if not scaler_file_path.exists():
                        raise FileNotFoundError(f"Scaler file not found: {scaler_file_path}")
                    scaler_file = joblib.load(scaler_file_path)
                    y_pred_scaler = scaler_file.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    y_true_scaler = scaler_file.inverse_transform(y_true.reshape(-1, 1)).flatten()
                    mae_scaler = mean_absolute_error(y_true_scaler, y_pred_scaler)
                    rmse_scaler = np.sqrt(mean_squared_error(y_true_scaler, y_pred_scaler))
                    mae_from_scaler_file[labels_percentage].loc[idx, paradigm] = mae_scaler
                    rmse_from_scaler_file[labels_percentage].loc[idx, paradigm] = rmse_scaler

                    print(f"{idx} | MAE(val_df): {val_mae:.4f} | MAE(log): {mae_log:.4f} | MAE(scaler.pkl): {mae_scaler:.4f}")
                    print(f"{idx} | RMSE(val_df): {val_rmse:.4f} | RMSE(log): {rmse_log:.4f} | RMSE(scaler.pkl): {rmse_scaler:.4f}")
                except Exception as e:
                    print(f"❌ Error processing {idx}: {e}")

# === Save all ===
os.makedirs(destination_path, exist_ok=True)
for lp in labels_percentage_list:
    val_mae_file[lp].to_csv(destination_path / f'val_MAE_from_file_labels_{lp}.csv')
    val_rmse_file[lp].to_csv(destination_path / f'val_RMSE_from_file_labels_{lp}.csv')

    mae_from_log[lp].to_csv(destination_path / f'recomputed_MAE_from_log_labels_{lp}.csv')
    rmse_from_log[lp].to_csv(destination_path / f'recomputed_RMSE_from_log_labels_{lp}.csv')

    mae_from_scaler_file[lp].to_csv(destination_path / f'recomputed_MAE_from_scaler_file_labels_{lp}.csv')
    rmse_from_scaler_file[lp].to_csv(destination_path / f'recomputed_RMSE_from_scaler_file_labels_{lp}.csv')

    print(f'\n✅ Results saved for labels_percentage={lp}')
