#!/bin/bash

# Constants (unchanged)
images_dir=${images_dir:-/Ironman/scratch/Andrea/data_from_bernadette/AGE_prediction/09_10_2023/AgePred_part1-2-3.nii.gz}
tabular_dir=${tabular_dir:-/Ironman/scratch/Andrea/data_from_bernadette/AGE_prediction/09_10_2023/NF_Andrea_part1-2-3.csv}
resize_shape=${resize_shape:-224}
train_classes_percentage_values=${train_classes_percentage_values:-None}
num_classes=${num_classes:-2}
balanced_val_set=${balanced_val_set:-True}
cross_val_folds=${cross_val_folds:-1}
normalization=${normalization:-'standardization'}
augmentation_rate=${augmentation_rate:-0.9}
projector=${projector:-1024-1024}
num_workers=${num_workers:-4}
device=${device:-cuda:0}

# Finetuning
finetune_epochs=${finetune_epochs:-1000}
finetune_min_epochs=${finetune_min_epochs:-500}
finetune_patience=${finetune_patience:-50}
finetune_batch_size=${finetune_batch_size:-512}
finetune_val_batch_size=${finetune_val_batch_size:-512}
finetune_head_lr=${finetune_head_lr:-0.001}
finetune_weight_decay=${finetune_weight_decay:-1e-6}
finetune_weighted_loss=${finetune_weighted_loss:-1}
finetune_pretrained_path=${finetune_pretrained_path:-exp}
finetune_lr_backbone=${finetune_lr_backbone:-0.0}
finetune_freeze_backbone=${finetune_freeze_backbone:-1}

# Paradigm-specific params
simim_bottleneck=${simim_bottleneck:-1}
simim_depth=${simim_depth:-12}
simim_mlp_ratio=${simim_mlp_ratio:-4}
simim_num_heads=${simim_num_heads:-6}
simim_emb_dim=${simim_emb_dim:-384}
simim_encoder_stride=${simim_encoder_stride:-16}
simim_in_chans=${simim_in_chans:-3}
simim_use_bn=${simim_use_bn:-True}
simim_patch_size=${simim_patch_size:-16}
simim_mask_patch_size=${simim_mask_patch_size:-32}
simim_mask_ratio=${simim_mask_ratio:-0.5}
simim_drop_path_rate=${simim_drop_path_rate:-0.1}

# Base directory containing experiments
base_dir="/Ironman/scratch/Andrea/med-booster/" #"../REVISION1"


# Loop through each matching experiment folder
for EXPERIMENT_PATH in "$base_dir"/EXPERIMENTS_MRI_augm_21_11/EXPS/seed0/*/*/labels_percentage_*; do
    if [ -d "$EXPERIMENT_PATH" ]; then
        # Extract components from the path
        labels_percentage=$(basename "$EXPERIMENT_PATH" | cut -d'_' -f3)
        paradigm=$(basename "$(dirname "$EXPERIMENT_PATH")")
        dataset_name=$(basename "$(dirname "$(dirname "$EXPERIMENT_PATH")")")
        seed=$(basename "$(dirname "$(dirname "$(dirname "$EXPERIMENT_PATH")")")" | sed 's/seed//')
        backbone="deit"  # fixed since only "deit" is relevant here

        echo "Running: SEED=$seed | PARADIGM=$paradigm | DATASET=$dataset_name | PERC=$labels_percentage | BACKBONE=$backbone"

        CUDA_VISIBLE_DEVICES=2 python source/get_scaling.py \
            --paradigm ${paradigm} \
            --images_dir ${images_dir} \
            --tabular_dir ${tabular_dir} \
            --dataset_name ${dataset_name} \
            --train_classes_percentage_values ${train_classes_percentage_values} \
            --resize_shape ${resize_shape} \
            --num_classes ${num_classes} \
            --labels_percentage ${labels_percentage} \
            --balanced_val_set ${balanced_val_set} \
            --normalization ${normalization} \
            --cross_val_folds ${cross_val_folds} \
            --backbone ${backbone} \
            --pretrained_path exp \
            --exp-dir ${EXPERIMENT_PATH} \
            --epochs ${finetune_epochs} \
            --fine-tune-batch-size ${finetune_batch_size} \
            --val-batch-size ${finetune_val_batch_size} \
            --lr-backbone ${finetune_lr_backbone} \
            --lr-head ${finetune_head_lr} \
            --weight-decay ${finetune_weight_decay} \
            --freeze_backbone ${finetune_freeze_backbone} \
            --weighted_loss ${finetune_weighted_loss} \
            --num_workers ${num_workers} \
            --device ${device} \
            --simim_bottleneck ${simim_bottleneck} \
            --simim_depth ${simim_depth} \
            --simim_mlp_ratio ${simim_mlp_ratio} \
            --simim_num_heads ${simim_num_heads} \
            --simim_emb_dim ${simim_emb_dim} \
            --simim_encoder_stride ${simim_encoder_stride} \
            --simim_in_chans ${simim_in_chans} \
            --simim_use_bn ${simim_use_bn} \
            --simim_patch_size ${simim_patch_size} \
            --simim_mask_patch_size ${simim_mask_patch_size} \
            --simim_mask_ratio ${simim_mask_ratio} \
            --simim_drop_path_rate ${simim_drop_path_rate} \
            --patience ${finetune_patience} \
            --min_epochs ${finetune_min_epochs} \
            --seed ${seed} \
            > "${EXPERIMENT_PATH}/get_scaler.log" 2>&1
    fi
done
