
original_folder_name=EXPERIMENTS_2023_11_13_ARMAND_thesis_correct
mkdir -p 'RESULTS_'${original_folder_name};

# Define the directory where you want to start searching
directory=EXPERIMENTS_2023_11_13_ARMAND_thesis_correct

rsync -av --exclude '*batch*' --exclude  '*.png' --exclude  '*.pth' --exclude  '*.npy' EXPERIMENTS_2023_11_13_ARMAND_thesis_correct 'RESULTS_'${original_folder_name}

# # Use find to search for files and directories containing 'val' or 'fine'
# # and store the results in an array
# mapfile -t matches < <(find "$directory" -type f -o -type d | grep -E 'val|fine')

# # Loop through the matches array and delete each file or directory
# for item in "${matches[@]}"; do
#     if [ -f "$item" ]; then
#         rm -f "$item"  # Delete files
#         echo "Deleted file: $item"
#     elif [ -d "$item" ]; then
#         rm -rf "$item"  # Delete directories
#         echo "Deleted directory: $item"
#     fi
# done
