from pathlib import Path
import os
import pandas as pd
import numpy as np

# --------- SETTINGS ---------
seeds = range(5)
paradigms = ['medbooster', 'supervised']
metric = 'RMSE'  # or 'MAE'
folds = range(1)
labels_percentage_list = [100, 1]

source_folder_path_all_experiments = Path(
    '/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_16_LARS_long_no_augm'
)
destination_folder_path = source_folder_path_all_experiments / 'results_AGE'
os.makedirs(destination_folder_path, exist_ok=True)

# --------- FUNCTION: EXTRACT MEAN & STD ---------
def extract_mean_std_from_log(log_path):
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "before scaling train set age mean" in line:
                    parts = line.strip().split("mean")[1].split(",")
                    mean = float(parts[0].strip())
                    std = float(parts[1].split("std")[1].strip())
                    return mean, std
        raise ValueError(f"Mean/std not found in {log_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to parse log {log_path}: {e}")

# --------- INITIALIZE RESULTS ---------
indexes = [f'seed_{seed}_fold_{fold}' for seed in seeds for fold in folds]
data = {paradigm: [0.0] * len(indexes) for paradigm in paradigms}
results_collected_metric = {
    labels_percentage: pd.DataFrame(data, index=indexes)
    for labels_percentage in labels_percentage_list
}

# --------- MAIN LOOP ---------
for seed in seeds:
    for dataset_name in ['AGE']:
        for paradigm in paradigms:
            print(f"\n🔍 Evaluating: {paradigm}")
            for labels_percentage in labels_percentage_list:
                for k_fold in folds:
                    folder_name = f'seed{seed}'
                    base_path = (
                        source_folder_path_all_experiments
                        / folder_name
                        / dataset_name
                        / paradigm
                        / f'labels_percentage_{labels_percentage}'
                        / f'fold_{k_fold}'
                    )

                    file_path_finetuning = base_path / 'finetuning_metrics' / 'finetuning_df_metrics.csv'
                    file_path_validation = base_path / 'val_metrics' / 'val_df_metrics.csv'

                    try:
                        df_finetune = pd.read_csv(file_path_finetuning)
                        df_val = pd.read_csv(file_path_validation)

                        # Get best epoch from fine-tuning loss
                        min_row = df_finetune.loc[df_finetune['fine_tuning_loss'].idxmin()]
                        epoch_number = int(min_row['epoch'])
                        print(f"  📌 Best epoch based on fine_tuning_loss: {epoch_number}")

                        val_row = df_val[df_val['epoch'] == epoch_number]
                        if val_row.empty:
                            print(f"⚠️ Epoch {epoch_number} not found in df_val, skipping...")
                            continue
                        val_row = val_row.iloc[0]

                        saved_mae = val_row['val_MAE_loss']

                        output_folder = base_path / 'valdiation_outputs_and_targets'
                        preds_path = output_folder / f'y_predicted_validation_epoch{epoch_number}.npy'
                        targets_path = output_folder / f'y_target_validation_epoch{epoch_number}.npy'

                        if not preds_path.exists() or not targets_path.exists():
                            print(f"⚠️ Missing prediction or target files for epoch {epoch_number}, skipping...")
                            continue

                        y_pred = np.load(preds_path).astype(np.float64).flatten()
                        y_true = np.load(targets_path).astype(np.float64).flatten()

                        # --------- ARRAY CHECKS ---------
                        print("    🧪 Array checks:")
                        print(f"      y_pred: shape={y_pred.shape}, mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")
                        print(f"      y_true: shape={y_true.shape}, mean={y_true.mean():.4f}, std={y_true.std():.4f}")
                        if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
                            print(f"⚠️ Non-finite values in y_pred or y_true, skipping...")
                            continue
                        if y_pred.shape != y_true.shape:
                            print(f"⚠️ Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}, skipping...")
                            continue

                        # 🔧 LOG PATH FIXED (one level above fold_0)
                        log_path = base_path.parent / 'finetuning_output.log'
                        try:
                            mean, std = extract_mean_std_from_log(log_path)
                        except Exception as e:
                            print(f"⚠️ {e}")
                            continue

                        # Rescale
                        y_pred_rescaled = y_pred * std + mean
                        y_true_rescaled = y_true * std + mean

                        # Recompute MAE
                        mae_val = np.mean(np.abs(y_pred_rescaled - y_true_rescaled))

                        # Compute RMSE or use MAE directly
                        if metric == 'RMSE':
                            val_best_metric = np.sqrt(np.mean((y_pred_rescaled - y_true_rescaled) ** 2))
                        elif metric == 'MAE':
                            val_best_metric = mae_val
                        else:
                            raise ValueError(f"Unsupported metric: {metric}")

                        # Print comparison
                        print(f"    📈 Saved MAE        = {saved_mae:.4f}")
                        print(f"    📉 Recomputed MAE   = {mae_val:.4f}")
                        if metric == 'RMSE':
                            print(f"    ✅ RMSE             = {val_best_metric:.4f}")

                        # Store result
                        results_collected_metric[labels_percentage].loc[
                            f'seed_{seed}_fold_{k_fold}', paradigm
                        ] = val_best_metric

                    except Exception as e:
                        print(f"⚠️ Error in seed {seed}, paradigm {paradigm}, labels {labels_percentage}, fold {k_fold}: {e}")
                        continue

# --------- SAVE RESULTS ---------
for labels_percentage in labels_percentage_list:
    output_file = destination_folder_path / (
        f'results_collected_{metric}_seeds_{list(seeds)}_folds_{list(folds)}_labels_percentage_{labels_percentage}_paradigms_{paradigms}.csv'
    )
    results_collected_metric[labels_percentage].to_csv(output_file, index=True)
    print(f"\n✅ {metric} for labels_percentage={labels_percentage}:\n{results_collected_metric[labels_percentage]}")
