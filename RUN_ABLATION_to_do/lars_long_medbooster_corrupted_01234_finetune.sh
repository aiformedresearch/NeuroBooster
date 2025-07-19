#!/bin/bash

# Define arrays for values to sweep over
seeds=(0 1 2 3 4)
labels_percentages=(100 1)
paradigms=(medbooster_corrupted)
datasets=(AGE)
backbones=(resnet34)

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

# Pretraining
pretrain_epochs=${pretrain_epochs:-2500}
pretrain_min_epochs=${pretrain_min_epochs:-100}
pretrain_patience=${pretrain_patience:-100}
pretrain_batch_size=${pretrain_batch_size:-256}
pretrain_optim=${pretrain_optim:-LARS}
pretrain_base_lr=${pretrain_base_lr:-0.05} #5e-4 fixed for beit or deit backbone!!
pretrain_weight_decay=${pretrain_weight_decay:-1e-6}
pretrain_weighted_loss=${pretrain_weighted_loss:-True} # in our experiments the pre-training was performed solely on the regression task so this parameter was not used

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
vicreg_sim_coeff=${vicreg_sim_coeff:-25.0}
vicreg_std_coeff=${vicreg_std_coeff:-25.0}
vicreg_cov_coeff=${vicreg_cov_coeff:-1.0}
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

# Iterate over combinations
for seed in "${seeds[@]}"; do
  for paradigm in "${paradigms[@]}"; do
    for dataset_name in "${datasets[@]}"; do
      for backbone in "${backbones[@]}"; do
        for labels_percentage in "${labels_percentages[@]}"; do
          EXPERIMENT_FOLDER_NAME=../REVISION1/EXPERIMENTS_ABLATION_2025_07_10_LARS_long_medbooster_corrupted_resnet/seed${seed}/${dataset_name}/${paradigm}/labels_percentage_${labels_percentage}
          FOLD0_FOLDER="${EXPERIMENT_FOLDER_NAME}/fold_0"
          pretrain_DONE_FILE="${FOLD0_FOLDER}/pretraining_done.txt"
          finetune_DONE_FILE="${FOLD0_FOLDER}/finetuning_ablation_done.txt"
          mkdir -p "$FOLD0_FOLDER"
          cp -r "$(realpath ./source/..)" "${EXPERIMENT_FOLDER_NAME}/repo_used_for_this_exp"

          echo "Running: SEED=$seed | PARADIGM=$paradigm | DATASET=$dataset_name | PERC=$labels_percentage | BACKBONE=$backbone"

          # if [[ ( "$labels_percentage" -eq 100 || "$paradigm" == "supervised" ) && ! -f "$pretrain_DONE_FILE" ]]; then

          #   CUDA_VISIBLE_DEVICES=1 python source/pretraining_deit_LARS.py \
          #     --paradigm ${paradigm} \
          #     --labels_percentage ${labels_percentage} \
          #     --images_dir ${images_dir} \
          #     --tabular_dir ${tabular_dir} \
          #     --dataset_name ${dataset_name} \
          #     --seed ${seed} \
          #     --exp-dir ${EXPERIMENT_FOLDER_NAME} \
          #     --backbone ${backbone} \
          #     --projector ${projector} \
          #     --batch-size ${pretrain_batch_size} \
          #     --cross_val_folds ${cross_val_folds} \
          #     --device ${device} \
          #     --base_lr ${pretrain_base_lr} \
          #     --optim ${pretrain_optim} \
          #     --min_epochs ${pretrain_min_epochs} \
          #     --patience ${pretrain_patience} \
          #     --num_workers ${num_workers} \
          #     --epochs ${pretrain_epochs} \
          #     --resize_shape ${resize_shape} \
          #     --normalization ${normalization} \
          #     --augmentation_rate ${augmentation_rate} \
          #     --vicreg_sim_coeff ${vicreg_sim_coeff} \
          #     --vicreg_std_coeff ${vicreg_std_coeff} \
          #     --vicreg_cov_coeff ${vicreg_cov_coeff} \
          #     --simim_bottleneck ${simim_bottleneck} \
          #     --simim_depth ${simim_depth} \
          #     --simim_mlp_ratio ${simim_mlp_ratio} \
          #     --simim_num_heads ${simim_num_heads} \
          #     --simim_emb_dim ${simim_emb_dim} \
          #     --simim_encoder_stride ${simim_encoder_stride} \
          #     --simim_in_chans ${simim_in_chans} \
          #     --simim_use_bn ${simim_use_bn} \
          #     --simim_patch_size ${simim_patch_size} \
          #     --simim_mask_patch_size ${simim_mask_patch_size} \
          #     --simim_mask_ratio ${simim_mask_ratio} \
          #     --simim_drop_path_rate ${simim_drop_path_rate} \
          #     --weight-decay ${pretrain_weight_decay} \
          #     --weighted_loss ${pretrain_weighted_loss} \
          #     > ${EXPERIMENT_FOLDER_NAME}/training_output.log 2>&1 &
          # else
          #   echo "⏭ Skipping training (already done or not needed)"
          # fi

          if [[ ! -f "$finetune_DONE_FILE" ]]; then
            CUDA_VISIBLE_DEVICES=2 python source/fine_tune_evaluate_deit.py \
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
              --exp-dir ${EXPERIMENT_FOLDER_NAME} \
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
              > ${EXPERIMENT_FOLDER_NAME}/finetuning_output.log 2>&1 &
          
          else
            echo "⏭ Skipping finetuning (already done)"
          fi

        done
      done
    done
  done
done
