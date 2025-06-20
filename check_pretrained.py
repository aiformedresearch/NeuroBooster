#!/usr/bin/env python3

import os
from collections import defaultdict

# Base directory
base_dir = "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_11_ADAMW_short"

# Seeds and labels percentages to check
seeds = range(5)
labels_percentages = [100, 1]

# Paradigms to organize by
paradigms = ["supervised", "medbooster", "vicreg", "mae"]

# Dictionaries to hold found and missing fold_0 folders per paradigm
found_folders = defaultdict(list)
missing_folders = defaultdict(list)

# Systematically check each combination
for seed in seeds:
    for paradigm in paradigms:
        for perc in labels_percentages:
            # Skip: paradigms that are not supervised and labels_percentage != 100
            if paradigm != "supervised" and perc != 100:
                continue

            # Construct expected fold_0 path
            fold_0_path = os.path.join(
                base_dir,
                f"seed{seed}",
                "AGE",
                paradigm,
                f"labels_percentage_{perc}",
                "fold_0"
            )
            pretraining_done_file = os.path.join(fold_0_path, "pretraining_done.txt")

            if os.path.isfile(pretraining_done_file):
                found_folders[paradigm].append(fold_0_path)
            else:
                missing_folders[paradigm].append(fold_0_path)

# 🟢 First: Print found
print("\n========== FOUND `pretraining_done.txt` ==========")
for paradigm in paradigms:
    paths = found_folders.get(paradigm, [])
    print(f"\nParadigm: {paradigm}")
    print(f"Found pretraining_done.txt count: {len(paths)}")
    for path in sorted(paths):
        print(path)

# 🔴 Then: Print missing
print("\n========== MISSING `pretraining_done.txt` ==========")
for paradigm in paradigms:
    paths = missing_folders.get(paradigm, [])
    print(f"\nParadigm: {paradigm}")
    print(f"Missing pretraining_done.txt count: {len(paths)}")
    for path in sorted(paths):
        print(path)
