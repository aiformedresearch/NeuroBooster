import os
from collections import defaultdict

# Base directory
base_dir = "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_2025_05_23_LARS"

# Paradigms to consider
paradigms = ['vicreg', 'made', 'supervised', 'simim', 'medbooster']

# Seeds range
seeds = range(30)

# Label percentages
label_percentages = ['100', '1']

# Dictionary to hold paths clustered by paradigm
paths_by_paradigm = defaultdict(list)

# Walk through the base directory
for paradigm in paradigms:
    for seed in seeds:
        for label_pct in label_percentages:
            search_path = os.path.join(base_dir, f"seed{seed}", "AGE", paradigm, f"labels_percentage_{label_pct}")
            finetuning_done_path = os.path.join(search_path, "fold_0", "finetuning_done.txt")
            if os.path.isfile(finetuning_done_path):
                paths_by_paradigm[paradigm].append(finetuning_done_path)

# Print out the results
for paradigm, paths in paths_by_paradigm.items():
    print(f"\nParadigm: {paradigm} ({len(paths)} found)")
    for path in paths:
        print(path)
