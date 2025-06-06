import numpy as np
import os
from pathlib import Path
import pandas as pd

def create_raw_comparison_data(experiment_configs: list, dataset_name: str = 'AGE', metrics: list = ['MAE'],
                               labels_percentage_list: list = [100, 1]) -> pd.DataFrame:
    """
    Collects raw mean and standard deviation for specified metrics from multiple experiment folders.

    Args:
        experiment_configs (list): A list of dictionaries, where each dictionary defines
                                   an experiment's configuration, e.g.,
                                   {
                                       'name': 'Experiment_1',
                                       'folder_root': Path('/path/to/exp1'),
                                       'paradigms_in_file': ['supervised', 'medbooster', ...],
                                       'paradigms_display_names': {'supervised': 'SL', 'medbooster': 'MB', ...},
                                       'seeds_in_file_str': 'range(0, 5)', # String as it appears in filename
                                       'folds_in_file_str': 'range(0, 1)',
                                       'filter_seeds_to_range': range(5) # The actual seeds to extract (e.g., 0-4)
                                   }
        dataset_name (str): Name of the dataset (e.g., 'AGE').
        metrics (list): List of metrics to collect (e.g., ['MAE']).
        labels_percentage_list (list): List of label percentages to process.

    Returns:
        pd.DataFrame: A DataFrame containing collected mean and standard deviation for all experiments.
    """

    all_results = []

    # Iterate through each experiment configuration provided
    for config in experiment_configs:
        experiment_name = config['name']
        experiment_folder_root = config['folder_root']
        paradigms_in_file = config['paradigms_in_file']
        paradigms_display_names = config['paradigms_display_names']
        seeds_in_file_str = config['seeds_in_file_str']
        folds_in_file_str = config['folds_in_file_str']
        filter_seeds_to_range = config['filter_seeds_to_range'] # Use this for .iloc filtering

        print(f"Processing experiment: {experiment_name}")

        source_folder_path = Path(experiment_folder_root) / f'results_{dataset_name}'

        if not source_folder_path.exists():
            print(f"Warning: Results folder not found for experiment '{experiment_name}' at {source_folder_path}. Skipping.")
            continue

        for labels_percentage in labels_percentage_list:
            for metric in metrics:
                # Construct the exact filename based on the experiment's configuration
                results_file_name = (
                    f'results_collected_{metric}'
                    f'_seeds_{seeds_in_file_str}'
                    f'_folds_{folds_in_file_str}'
                    f'_labels_percentage_{labels_percentage}'
                    f'_paradigms_{str(paradigms_in_file)}.csv' # Ensure paradigms list is stringified correctly
                )
                file_path = source_folder_path / results_file_name

                if not file_path.exists():
                    print(f"Warning: File not found: {file_path}. Skipping this combination.")
                    continue

                try:
                    df = pd.read_csv(file_path)
                    df = df.drop('Unnamed: 0', axis=1, errors='ignore')

                    # Filter rows by specified seeds if a range is provided
                    if filter_seeds_to_range is not None:
                        # Assumes each row corresponds to a seed in order
                        df = df.iloc[list(filter_seeds_to_range), :]
                        if df.empty:
                            print(f"Warning: After filtering, no data for seeds {list(filter_seeds_to_range)} in {file_path}. Skipping.")
                            continue

                    # Rename columns to their display names for this specific experiment
                    # Use .get() with a default to handle cases where a column might not exist or isn't mapped
                    current_df_columns = df.columns.tolist() # Get current column names
                    df.columns = [paradigms_display_names.get(col, col) for col in current_df_columns]

                    # Collect results for each paradigm in this experiment
                    for original_paradigm_name in paradigms_in_file:
                        display_name = paradigms_display_names.get(original_paradigm_name, original_paradigm_name)
                        if display_name in df.columns:
                            mean_val = df[display_name].mean()
                            std_val = df[display_name].std()

                            all_results.append({
                                'Experiment': experiment_name,
                                'Labels Percentage': labels_percentage,
                                'Metric': metric,
                                'Paradigm': display_name, # Use the display name here
                                'Mean': mean_val,
                                'Std': std_val
                            })
                        else:
                            print(f"Warning: Paradigm '{display_name}' (originally '{original_paradigm_name}') not found in {file_path}.")

                except Exception as e:
                    print(f"Error reading or processing {file_path}: {e}")

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    # --- Configuration for all experiments ---
    # Define experiment configurations in a list of dictionaries

    # Configuration for OLD Experiment
    old_experiment_folder = Path('/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long')
    old_paradigms = ['supervised', 'medbooster', 'vicreg', 'simim', 'mae']
    old_paradigms_names = {'supervised': 'SL', 'medbooster': 'MB', 'vicreg': 'VICReg', 'simim': 'SimMIM', 'baseline': 'baseline', 'mae': 'MAE'}
    old_seeds_range = range(5)
    old_folds_range = range(1)

    old_experiment_config = {
        'name': 'LARS_long', # A descriptive name for this experiment
        'folder_root': old_experiment_folder,
        'paradigms_in_file': old_paradigms,
        'paradigms_display_names': old_paradigms_names,
        'seeds_in_file_str': str(old_seeds_range),
        'folds_in_file_str': str(old_folds_range),
        'filter_seeds_to_range': old_seeds_range # We want all 5 seeds from this one
    }

    # Configuration for NEW Experiment
    new_experiment_folder = Path('/Ironman/scratch/Andrea/med-booster/EXPERIMENTS_MRI_augm_21_11')
    new_paradigms_in_file = ['supervised', 'neurobooster', 'vicreg', 'simim']
    # *** IMPORTANT CHANGE HERE ***
    # 'neurobooster' from the new experiment is now mapped to 'MB'
    new_paradigms_display_names = {'supervised': 'SL', 'neurobooster': 'MB', 'vicreg': 'VICReg', 'simim': 'SimMIM'}
    # The actual strings as found in the new experiment's filenames
    new_exp_seeds_file_str_in_filename = 'range(0, 30)'
    new_exp_folds_file_str_in_filename = 'range(0, 1)'
    target_seeds_for_new_exp = range(5) # We only want to use seeds 0-4 from this data

    new_experiment_config = {
        'name': 'MRI_augm_21_11', # A descriptive name for this new experiment
        'folder_root': new_experiment_folder,
        'paradigms_in_file': new_paradigms_in_file,
        'paradigms_display_names': new_paradigms_display_names,
        'seeds_in_file_str': new_exp_seeds_file_str_in_filename,
        'folds_in_file_str': new_exp_folds_file_str_in_filename,
        'filter_seeds_to_range': target_seeds_for_new_exp # Filter to seeds 0-4
    }

    # Add other experiments if needed (e.g., the ADAMW_short and LARS_short from your original script)
    # You would create similar config dictionaries for them.
    # For example:
    # experiment_root_folder2 = Path('/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_ADAMW_short')
    # adamw_short_config = {
    #     'name': 'ADAMW_short',
    #     'folder_root': experiment_root_folder2,
    #     'paradigms_in_file': old_paradigms, # Assuming same paradigms as old experiment
    #     'paradigms_display_names': old_paradigms_names,
    #     'seeds_in_file_str': str(old_seeds_range),
    #     'folds_in_file_str': str(old_folds_range),
    #     'filter_seeds_to_range': old_seeds_range
    # }

    # List of all experiment configurations to compare
    experiment_configs_to_compare = [
        old_experiment_config,
        new_experiment_config,
        # adamw_short_config, # Uncomment if you want to include ADAMW_short
    ]

    # Parameters for data collection (common to all experiments being collected)
    dataset_name = 'AGE'
    metrics = ['MAE']
    labels_percentage_list = [100, 1]

    # --- Data Collection ---
    raw_comparison_df = create_raw_comparison_data(
        experiment_configs=experiment_configs_to_compare,
        dataset_name=dataset_name,
        metrics=metrics,
        labels_percentage_list=labels_percentage_list
    )

    # --- Table Formatting and Output ---
    if raw_comparison_df.empty:
        print("\n❌ No results were found to create any comparison tables. Please verify your experiment paths and file naming conventions.")
    else:
        # Convert 'Mean' and 'Std' columns to numeric, coercing errors and filling NaN std with 0 for consistent display
        raw_comparison_df['Mean'] = pd.to_numeric(raw_comparison_df['Mean'], errors='coerce')
        raw_comparison_df['Std'] = pd.to_numeric(raw_comparison_df['Std'], errors='coerce').fillna(0) # Fill NaN std with 0 for display

        # Define the output directory
        output_dir = Path('/Ironman/scratch/Andrea/med-booster/REVISION1/COMBINED_COMPARISON_RESULTS')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all unique paradigms found across all experiments to ensure consistent row order
        # This will now correctly group 'medbooster' and 'neurobooster' under 'MB'
        all_paradigms_found = sorted(raw_comparison_df['Paradigm'].unique())
        # Manual order for specific display control, e.g.:
        # all_paradigms_found = ['SL', 'MB', 'VICReg', 'SimMIM', 'MAE']


        for labels_percentage in labels_percentage_list:
            print(f"\nGenerating table for Labels Percentage: {labels_percentage}%")
            
            # Filter data for the current labels percentage
            df_filtered = raw_comparison_df[raw_comparison_df['Labels Percentage'] == labels_percentage].copy()

            if df_filtered.empty:
                print(f"No data found for Labels Percentage {labels_percentage}%. Skipping table generation for this percentage.")
                continue

            # Combine Mean and Std into a single formatted string (3 decimal places)
            df_filtered['Mean_Std_Combined'] = df_filtered.apply(
                lambda row: f"{row['Mean']:.3f} ({row['Std']:.3f})" if pd.notna(row['Mean']) else 'N/A', axis=1
            )

            # Pivot the table:
            # - Rows: Paradigm
            # - Columns: Experiment Names
            # - Values: The combined "Mean (Std)" string
            comparison_table_pivot = df_filtered.pivot_table(
                index='Paradigm',
                columns='Experiment',
                values='Mean_Std_Combined',
                aggfunc='first' # Use 'first' because 'Mean_Std_Combined' is already pre-aggregated
            )

            # Reindex to ensure all paradigms are present in a consistent order
            # Fill missing rows (paradigms not present in a specific percentage) with N/A
            comparison_table_pivot = comparison_table_pivot.reindex(all_paradigms_found, fill_value='N/A')

            # Fill any missing values (e.g., if an experiment didn't have data for a specific combination)
            comparison_table_pivot = comparison_table_pivot.fillna('N/A')

            # Define output paths for the specific percentage table
            file_suffix = f"labels_{labels_percentage}percent_combined"
            output_csv_path = output_dir / f'experiment_comparison_table_{file_suffix}.csv'
            output_txt_path = output_dir / f'experiment_comparison_table_{file_suffix}.txt'

            # Save the formatted table to CSV
            comparison_table_pivot.to_csv(output_csv_path)
            print(f"✅ Table for {labels_percentage}% saved to: {output_csv_path}")

            # Also print to a formatted text file for easy viewing
            with open(output_txt_path, 'w') as f:
                f.write(comparison_table_pivot.to_string())
            print(f"✅ Table for {labels_percentage}% also saved to: {output_txt_path}")