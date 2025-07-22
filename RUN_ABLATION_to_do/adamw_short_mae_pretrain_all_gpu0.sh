# Loop over seeds in batches of 4
for ((i=0; i<=14; i+=4)); do
  batch=("${i}" "$((i+1))" "$((i+2))" "$((i+3))")

  for seed in "${batch[@]}"; do
    # Skip if seed > 29 (to handle edge case when not divisible by 4)
    if [ "$seed" -gt 29 ]; then
      continue
    fi

    for paradigm in "${paradigms[@]}"; do
      for dataset_name in "${datasets[@]}"; do
        for backbone in "${backbones[@]}"; do
          for labels_percentage in "${labels_percentages[@]}"; do
            EXPERIMENT_FOLDER_NAME=../REVISION1/EXPERIMENTS_ABLATION_2025_07_20_ADAMW_short/seed${seed}/${dataset_name}/${paradigm}/labels_percentage_${labels_percentage}
            FOLD0_FOLDER="${EXPERIMENT_FOLDER_NAME}/fold_0"
            pretrain_DONE_FILE="${FOLD0_FOLDER}/pretraining_done.txt"
            mkdir -p "$FOLD0_FOLDER"

            echo "Running: SEED=$seed | PARADIGM=$paradigm | DATASET=$dataset_name | PERC=$labels_percentage | BACKBONE=$backbone"

            if [[ ( "$labels_percentage" -eq 100 || "$paradigm" == "supervised" ) && ! -f "$pretrain_DONE_FILE" ]]; then
              CUDA_VISIBLE_DEVICES=0 python source/pretraining_deit_SIMIMOPT.py \
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
                > ${EXPERIMENT_FOLDER_NAME}/training_output.log 2>&1 &
              sleep 30
            else
              echo "⏭ Skipping training for SEED=$seed"
            fi
          done
        done
      done
    done
  done

  # Wait for all background processes in this batch to finish
  wait
  echo "✅ Finished batch starting with seed $i"
done
