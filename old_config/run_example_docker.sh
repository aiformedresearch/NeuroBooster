#!/bin/bash

# docker run --rm --gpus all \
#     --shm-size=8g \
#     -e images_dir='/home/andreaespis/diciotti/data/AGE_prediction/09_10_2023/AgePred_part1-2-3.nii.gz' \
#     -e tabular_dir='/home/andreaespis/diciotti/data/AGE_prediction/09_10_2023/AgePred_part1-2-3.nii.gz' \
#     -e EXPERIMENT_FOLDER_NAME=/Ironman/scratch/Andrea/med-booster/EXPERIMENT_DEBUG_EXAMPLE_AGE_1 \
#     -v /path/to/data:/app/data \
#     -v /path/to/exp_folder:/app/exp_folder \
#     med_booster_image

# images_dir='/home/andreaespis/diciotti/data/AGE_prediction/09_10_2023/AgePred_part1-2-3.nii.gz' tabular_dir='/home/andreaespis/diciotti/data/AGE_prediction/09_10_2023/NF_Andrea_part1-2-3.csv' bash /home/andreaespis/diciotti/andrea/med-booster/MAIN/run_example.sh

# Assign default values if environment variables are not set
EXPERIMENT_FOLDER_NAME=${EXPERIMENT_FOLDER_NAME:-./EXPERIMENT_DEBUG_EXAMPLE_AGE_1/}
paradigm=${paradigm:-supervised} # choices: supervised, medbooster, vicreg, simim

# Data
images_dir=${images_dir:-path/to/folder/containing/images/}
tabular_dir=${tabular_dir:-/path/to/folder/containing/tabular/data/}
labels_percentage=${labels_percentage:-100}
resize_shape=${resize_shape:-224}
dataset_name=${dataset_name:-AGE}
train_classes_percentage_values=${train_classes_percentage_values:-None}
num_classes=${num_classes:-2}
balanced_val_set=${balanced_val_set:-True}
cross_val_folds=${cross_val_folds:-1} # 1 for bootstrap validation schema
normalization=${normalization:-'standardization'}
augmentation_rate=${augmentation_rate:-0.9}

# Model architecture
projector=${projector:-1024-1024}
backbone=${backbone:-resnet34} # beit_small for simim paradigm

# General
seed=${seed:-0}
num_workers=${num_workers:-4}
device=${device:-cuda:0}


# Pretraining
pretrain_epochs=${pretrain_epochs:-2}
pretrain_min_epochs=${pretrain_min_epochs:-1}
pretrain_patience=${pretrain_patience:-1}
pretrain_batch_size=${pretrain_batch_size:-256}
pretrain_optim=${pretrain_optim:-LARS}
pretrain_base_lr=${pretrain_base_lr:-0.05}
pretrain_weight_decay=${pretrain_weight_decay:-1e-6}
pretrain_weighted_loss=${pretrain_weighted_loss:-True} # in our experiments the pre-training was performed solely on the regression task so this parameter was not used

# Finetuning
finetune_epochs=${finetune_epochs:-2}
finetune_min_epochs=${finetune_min_epochs:-1}
finetune_patience=${finetune_patience:-1}
finetune_batch_size=${finetune_batch_size:-512}
finetune_val_batch_size=${finetune_val_batch_size:-512}
finetune_head_lr=${finetune_head_lr:-0.001}
finetune_weight_decay=${finetune_weight_decay:-1e-6}
finetune_weighted_loss=${finetune_weighted_loss:-1} 
finetune_pretrained_path=${finetune_pretrained_path:-exp}
finetune_lr_backbone=${finetune_lr_backbone:-0.0}
finetune_freeze_backbone=${finetune_freeze_backbone:-1}

# Paradigm-specific parameters for pretraining
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

# Print the values to verify
echo "Experiment folder name: ${EXPERIMENT_FOLDER_NAME}"
echo "Paradigm: ${paradigm}"
echo "Backbone: ${backbone}"
echo "Images directory: ${images_dir}"
echo "Tabular directory: ${tabular_dir}"
echo "Labels percentage: ${labels_percentage}"
echo "Resize shape: ${resize_shape}"
echo "Dataset name: ${dataset_name}"
echo "Train classes percentage values: ${train_classes_percentage_values}"
echo "Number of classes: ${num_classes}"
echo "Balanced validation set: ${balanced_val_set}"
echo "Normalization: ${normalization}"
echo "Cross validation folds: ${cross_val_folds}"
echo "Seed: ${seed}"
echo "Num workers: ${num_workers}"
echo "Device: ${device}"
echo "Pretrain epochs: ${pretrain_epochs}"
echo "Pretrain min epochs: ${pretrain_min_epochs}"
echo "Pretrain patience: ${pretrain_patience}"
echo "Pretrain batch size: ${pretrain_batch_size}"
echo "Pretrain optim: ${pretrain_optim}"
echo "Pretrain base LR: ${pretrain_base_lr}"
echo "Pretrain weight decay: ${pretrain_weight_decay}"
echo "Pretrain weighted loss: ${pretrain_weighted_loss}"
echo "Finetune epochs: ${finetune_epochs}"
echo "Finetune min epochs: ${finetune_min_epochs}"
echo "Finetune patience: ${finetune_patience}"
echo "Finetune batch size: ${finetune_batch_size}"
echo "Finetune val batch size: ${finetune_val_batch_size}"
echo "Finetune head LR: ${finetune_head_lr}"
echo "Finetune weight decay: ${finetune_weight_decay}"
echo "Finetune weighted loss: ${finetune_weighted_loss}"
echo "Finetune pretrained path: ${finetune_pretrained_path}"
echo "Finetune learning rate backbone: ${finetune_lr_backbone}"
echo "Finetune freeze backbone: ${finetune_freeze_backbone}"
echo "Augmentation rate: ${augmentation_rate}"
echo "Projector: ${projector}"
echo "VICReg Sim Coefficient: ${vicreg_sim_coeff}"
echo "VICReg Std Coefficient: ${vicreg_std_coeff}"
echo "VICReg Cov Coefficient: ${vicreg_cov_coeff}"
echo "SimIM Bottleneck: ${simim_bottleneck}"
echo "SimIM Depth: ${simim_depth}"
echo "SimIM MLP Ratio: ${simim_mlp_ratio}"
echo "SimIM Num Heads: ${simim_num_heads}"
echo "SimIM Embedding Dimension: ${simim_emb_dim}"
echo "SimIM Encoder Stride: ${simim_encoder_stride}"
echo "SimIM Input Channels: ${simim_in_chans}"
echo "SimIM Use Batch Normalization: ${simim_use_bn}"
echo "SimIM Patch Size: ${simim_patch_size}"
echo "SimIM Mask Patch Size: ${simim_mask_patch_size}"
echo "SimIM Mask Ratio: ${simim_mask_ratio}"
echo "SimIM Drop Path Rate: ${simim_drop_path_rate}"

mkdir -p ${EXPERIMENT_FOLDER_NAME};
echo "created folder ${EXPERIMENT_FOLDER_NAME}"

# Run pretraining script
CUDA_VISIBLE_DEVICES=0 python /app/source/pretraining.py \
    --paradigm ${paradigm} \
    --labels_percentage ${labels_percentage} \
    --images_dir ${images_dir} \
    --tabular_dir ${tabular_dir} \
    --dataset_name ${dataset_name} \
    --seed ${seed} \
    --exp-dir ${EXPERIMENT_FOLDER_NAME} \
    --backbone ${backbone} \
    --projector ${projector} \
    --batch-size ${pretrain_batch_size} \
    --cross_val_folds ${cross_val_folds} \
    --device ${device} \
    --base_lr ${pretrain_base_lr} \
    --optim ${pretrain_optim} \
    --min_epochs ${pretrain_min_epochs} \
    --patience ${pretrain_patience} \
    --num_workers ${num_workers} \
    --epochs ${pretrain_epochs} \
    --resize_shape ${resize_shape} \
    --normalization ${normalization} \
    --augmentation_rate ${augmentation_rate} \
    --vicreg_sim_coeff ${vicreg_sim_coeff} \
    --vicreg_std_coeff ${vicreg_std_coeff} \
    --vicreg_cov_coeff ${vicreg_cov_coeff} \
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
    --weight-decay ${pretrain_weight_decay} \
    --weighted_loss ${pretrain_weighted_loss} \
    > ${EXPERIMENT_FOLDER_NAME}/training_output.log 2>&1

if [ $? -ne 0 ]; then
    echo "Pretraining script failed. Check the log file for details."
    exit 1
fi

echo "Pretraining done. Log file saved to ${EXPERIMENT_FOLDER_NAME}/training_output.log"

# Run finetuning script
CUDA_VISIBLE_DEVICES=0 python /app/source/fine_tune_evaluate.py \
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
    --pretrained_path ${finetune_pretrained_path} \
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
    > ${EXPERIMENT_FOLDER_NAME}/finetuning_output.log 2>&1


if [ $? -ne 0 ]; then
    echo "Finetuning script failed. Check the log file for details."
    exit 1
fi

echo "Finetuning done. Log file saved to ${EXPERIMENT_FOLDER_NAME}/finetuning_output.log"