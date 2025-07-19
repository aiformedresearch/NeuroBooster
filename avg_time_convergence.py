import os
import json
from collections import defaultdict

def analyze_experiment(experiment_path):
    paradigm_epochs = defaultdict(list)
    paradigm_times = defaultdict(list)

    for root, dirs, files in os.walk(experiment_path):
        if 'pretraining_all_stats.txt' in files and 'labels_percentage_100/fold_0' in root:
            # Extract paradigm from path
            parts = root.split(os.sep)
            try:
                seed_idx = parts.index("AGE") - 1  # find the "seedX" index
                paradigm = parts[seed_idx + 2]
            except Exception as e:
                print(f"Skipping due to path error in {root}: {e}")
                continue

            stats_path = os.path.join(root, 'pretraining_all_stats.txt')
            try:
                with open(stats_path, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                    last_entry = json.loads(lines[-1])
                    last_epoch = last_entry.get("epoch", -1)
                    last_time = last_entry.get("time", -1)
                    final_epoch = last_epoch + 1  # Epochs are zero-indexed

                    # Ignore if final_epoch < 90
                    if final_epoch < 90:
                        print(f"Ignoring {stats_path} (final_epoch = {final_epoch})")
                        continue

                    if last_epoch != -1 and last_time != -1:
                        paradigm_epochs[paradigm].append(final_epoch)
                        paradigm_times[paradigm].append(last_time / 3600)  # convert seconds to hours
            except Exception as e:
                print(f"Failed to read {stats_path}: {e}")
                continue

    print(f"\nSummary of paradigms for exp {experiment_path} (ignoring runs with < 90 epochs):")
    for paradigm in sorted(paradigm_epochs.keys()):
        avg_epoch = sum(paradigm_epochs[paradigm]) / len(paradigm_epochs[paradigm])
        avg_time = sum(paradigm_times[paradigm]) / len(paradigm_times[paradigm])
        print(f"- {paradigm}: average max epochs = {avg_epoch:.2f}, average time = {avg_time:.2f} hours")

# Example usage
analyze_experiment('/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long')
analyze_experiment('/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_11_ADAMW_short')
