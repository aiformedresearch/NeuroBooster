# #!/bin/bash

# base_dir="/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_2025_05_22_1_simimoptimization"

# find "$base_dir" -type d -name "fold_0" | while read -r fold_path; do
#     if [ ! -f "$fold_path/pretraining_done.txt" ]; then
#         echo "$fold_path"
#     fi
# done


#!/bin/bash

base_dir="/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_2025_05_22_1_simimoptimization"

find "$base_dir" -type d -name "fold_0" | while read -r fold_path; do
    touch "${fold_path}/finetuning_done.txt"
    echo "✅ Created: ${fold_path}/finetuning_done.txt"
done
