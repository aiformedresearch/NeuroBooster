#!/bin/bash

for seed in {0..29}; do
  for dataset in AGE ADNI; do
    src="/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_07_08_LARS_long_simclr_resnet34/seed${seed}/${dataset}/simclr/"
    dst="/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11/EXPS/seed${seed}/${dataset}/simclr/"

    rsync -a --info=progress2 "$src" "$dst" &
  done
done

wait
