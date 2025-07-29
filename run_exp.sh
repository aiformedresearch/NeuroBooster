
# Check required env variables
if [ -z "$images_dir" ] || [ -z "$tabular_dir" ] || [ -z "$EXPERIMENT_FOLDER_NAME" ]; then
  echo "Error: One or more required environment variables are not set."
  echo "Please provide: images_dir, tabular_dir, and EXPERIMENT_FOLDER_NAME"
  exit 1
fi

# Define arrays for values to sweep over
seeds=(0)
labels_percentages=(100 1) # 100, 10, 1
paradigms=(supervised neurobooster simclr vicreg mae simim) # options
dataset_name=AGE # options: AGE, ADNI
backbones=(resnet34 deit)  # options: deit, resnet34, resnet34_3D
optimization_protocol=DEBUG # options: DEBUG, AdamW_short, LARS_long, LARS_short

CUDA_DEVICE=${CUDA_DEVICE:-0}

# Print current configuration
echo "Running experiment with:"
echo "images_dir: $images_dir"
echo "tabular_dir: $tabular_dir"
echo "EXPERIMENT_FOLDER_NAME: $EXPERIMENT_FOLDER_NAME"
echo "CUDA_DEVICE: $CUDA_DEVICE"

# Pretraining
if [ "$optimization_protocol" == "DEBUG" ]; then
    pretrain_epochs=${pretrain_epochs:-2}
    pretrain_min_epochs=${pretrain_min_epochs:-2}
    pretrain_patience=${pretrain_patience:-2}
    pretrain_batch_size=${pretrain_batch_size:-8}
    pretrain_optim=${pretrain_optim:-LARS}
    pretrain_base_lr=${pretrain_base_lr:-0.05}
    pretrain_weight_decay=${pretrain_weight_decay:-1e-6}

elif [ "$optimization_protocol" == "AdamW_short" ]; then
    pretrain_epochs=${pretrain_epochs:-300}
    pretrain_min_epochs=${pretrain_min_epochs:-200}
    pretrain_patience=${pretrain_patience:-50}
    pretrain_batch_size=${pretrain_batch_size:-256}
    pretrain_optim=${pretrain_optim:-ADAMW}
    pretrain_base_lr=${pretrain_base_lr:-5e-4}
    pretrain_weight_decay=${pretrain_weight_decay:-0.05}

elif [ "$optimization_protocol" == "LARS_long" ]; then
    pretrain_epochs=${pretrain_epochs:-2500}
    pretrain_min_epochs=${pretrain_min_epochs:-100}
    pretrain_patience=${pretrain_patience:-100}
    pretrain_batch_size=${pretrain_batch_size:-256}
    pretrain_optim=${pretrain_optim:-LARS}
    pretrain_base_lr=${pretrain_base_lr:-0.05} 
    pretrain_weight_decay=${pretrain_weight_decay:-1e-6}

elif [ "$optimization_protocol" == "LARS_short" ]; then
    pretrain_epochs=${pretrain_epochs:-300}
    pretrain_min_epochs=${pretrain_min_epochs:-200}
    pretrain_patience=${pretrain_patience:-50}
    pretrain_batch_size=${pretrain_batch_size:-256}
    pretrain_optim=${pretrain_optim:-LARS}
    pretrain_base_lr=${pretrain_base_lr:-0.05} 
    pretrain_weight_decay=${pretrain_weight_decay:-1e-6}
fi

# Finetuning
if [ "$optimization_protocol" == "DEBUG" ]; then
  finetune_epochs=${finetune_epochs:-2} 
  finetune_min_epochs=${finetune_min_epochs:-2}
  finetune_patience=${finetune_patience:-2}
else
  finetune_epochs=${finetune_epochs:-1000} 
  finetune_min_epochs=${finetune_min_epochs:-500}
  finetune_patience=${finetune_patience:-50}
fi

finetune_batch_size=${finetune_batch_size:-512}
finetune_val_batch_size=${finetune_val_batch_size:-512}
finetune_head_lr=${finetune_head_lr:-0.001}
finetune_weight_decay=${finetune_weight_decay:-1e-6}
finetune_weighted_loss=${finetune_weighted_loss:-1} 
finetune_pretrained_path=${finetune_pretrained_path:-exp}
finetune_lr_backbone=${finetune_lr_backbone:-0.0}
finetune_freeze_backbone=${finetune_freeze_backbone:-1}

resize_shape=${resize_shape:-224}
train_classes_percentage_values=${train_classes_percentage_values:-None}
num_classes=${num_classes:-2}
balanced_val_set=${balanced_val_set:-True}
normalization=${normalization:-'standardization'}
augmentation_rate=${augmentation_rate:-0.9}
projector=${projector:-1024-1024}
num_workers=${num_workers:-4}
device=${device:-cuda:0}
pretrain_weighted_loss=${pretrain_weighted_loss:-True} # in our experiments the pre-training was performed solely on the regression task so this parameter was not used

# Iterate over combinations
for seed in "${seeds[@]}"; do
  for paradigm in "${paradigms[@]}"; do
    for backbone in "${backbones[@]}"; do
      for labels_percentage in "${labels_percentages[@]}"; do 
        EXPERIMENT_PATH="../${EXPERIMENT_FOLDER_NAME}/seed${seed}/${dataset_name}/${paradigm}/labels_percentage_${labels_percentage}"
        pretrain_DONE_FILE="${EXPERIMENT_PATH}/pretraining_done.txt"
        finetune_DONE_FILE="${EXPERIMENT_PATH}/finetuning_done.txt"
        mkdir -p "$EXPERIMENT_PATH"
        cp -r "$(realpath ./source/..)" "${EXPERIMENT_PATH}/repo_used_for_this_exp"

        echo "Running: SEED=$seed | PARADIGM=$paradigm | DATASET=$dataset_name | PERC=$labels_percentage | BACKBONE=$backbone"

        if [[ ( "$labels_percentage" -eq 100 || "$paradigm" == "supervised" ) && ! -f "$pretrain_DONE_FILE" ]]; then

          CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python source/pretraining.py \
            --paradigm ${paradigm} \
            --labels_percentage ${labels_percentage} \
            --images_dir ${images_dir} \
            --tabular_dir ${tabular_dir} \
            --dataset_name ${dataset_name} \
            --seed ${seed} \
            --exp-dir ${EXPERIMENT_PATH} \
            --backbone ${backbone} \
            --projector ${projector} \
            --batch-size ${pretrain_batch_size} \
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
            --weight-decay ${pretrain_weight_decay} \
            --weighted_loss ${pretrain_weighted_loss} \
            > ${EXPERIMENT_PATH}/pretraining.log 2>&1
        else
          echo "⏭ Skipping training (already done or not needed)"
        fi

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python source/fine_tune_evaluate.py \
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
          --patience ${finetune_patience} \
          --min_epochs ${finetune_min_epochs} \
          --seed ${seed} \
          > ${EXPERIMENT_PATH}/fine_tune_evaluate.log 2>&1

      done
    done
  done
done
