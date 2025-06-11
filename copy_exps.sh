#!/bin/bash

# Base source and destination paths
SRC_BASE="/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_ADAMW_short"
DST_BASE="/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS"

# Datasets and seed range
DATASETS=("AGE" "ADNI")
SEED_START=0
SEED_END=29

for seed in $(seq $SEED_START $SEED_END); do
    for dataset in "${DATASETS[@]}"; do
        src_path="${SRC_BASE}/seed${seed}/${dataset}/mae"
        dst_path="${DST_BASE}/seed${seed}/${dataset}/"

        if [ -d "$src_path" ]; then
            echo "📦 Copying from: $src_path"
            echo "➡️  To: $dst_path"
            mkdir -p "$dst_path"
            cp -r "$src_path" "$dst_path"
        else
            echo "❌ Not found: $src_path"
        fi
    done
done
