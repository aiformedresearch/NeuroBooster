# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import matplotlib.gridspec as gridspec

# def extract_paradigm(path: Path) -> str:
#     parts = path.parts
#     if 'AGE' in parts:
#         idx = parts.index('AGE')
#         if idx + 1 < len(parts):
#             return parts[idx + 1]
#     return "Unknown"

# def plot_multiple_overlays(paths, save_path=None):
#     overlays = []
#     paradigms = []

#     paradigm_name_map = {
#         'supervised': 'SL',
#         'medbooster': 'NB',
#         'simclr': 'SimCLR',
#         'vicreg': 'VICReg',
#         'simim': 'SimMIM',
#         'mae': 'MAE'
#     }

#     desired_order = ['supervised', 'medbooster', 'simclr', 'vicreg', 'simim', 'mae']
#     path_dict = {extract_paradigm(Path(p)).lower(): p for p in paths}
#     sorted_paths = [path_dict[k] for k in desired_order if k in path_dict]

#     for path_str in sorted_paths:
#         path = Path(path_str)
#         mri_file = path / 'avg_image.npy'
#         sal_file = path / 'avg_saliency.npy'

#         if not (mri_file.exists() and sal_file.exists()):
#             print(f"⚠️ Missing files in: {path}")
#             continue

#         avg_image = np.load(mri_file)
#         avg_saliency = np.load(sal_file)
#         avg_saliency = (avg_saliency - avg_saliency.min()) / (avg_saliency.ptp() + 1e-8)

#         overlays.append((avg_image, avg_saliency))
#         raw_paradigm = extract_paradigm(path).lower()
#         paradigms.append(paradigm_name_map.get(raw_paradigm, raw_paradigm.upper()))

#     num_items = len(overlays)
#     if num_items == 0:
#         print("❌ No valid overlays found.")
#         return

#     fig_height_per_item = 2.4
#     fig = plt.figure(figsize=(4.4, num_items * fig_height_per_item))

#     # Use GridSpec with no internal spacing
#     gs = gridspec.GridSpec(num_items * 2, 2, figure=fig,
#                            height_ratios=[0.15, 1.0] * num_items,
#                            hspace=0.0, wspace=0.0)

#     for i, (image, saliency) in enumerate(overlays):
#         row_title = i * 2
#         row_images = i * 2 + 1

#         # Title axis: short row, text low in the box
#         ax_title = fig.add_subplot(gs[row_title, :])
#         ax_title.axis('off')
#         ax_title.text(0.5, 0.05, paradigms[i], transform=ax_title.transAxes,
#                       fontsize=12, fontweight='bold', ha='center', va='bottom')

#         # Saliency map
#         ax_sal = fig.add_subplot(gs[row_images, 0])
#         ax_sal.imshow(saliency, cmap='hot')
#         ax_sal.axis('off')

#         # Overlay
#         ax_overlay = fig.add_subplot(gs[row_images, 1])
#         ax_overlay.imshow(image, cmap='gray')
#         ax_overlay.imshow(saliency, cmap='hot', alpha=0.5)
#         ax_overlay.axis('off')

#     # This completely removes outer margins
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
#         print(f"✅ Saved comparison figure to {save_path}")
#     else:
#         plt.show()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_paradigm(path: Path) -> str:
    parts = path.parts
    if 'AGE' in parts:
        idx = parts.index('AGE')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "Unknown"

def plot_multiple_overlays(paths, save_path=None):
    overlays = []
    paradigms = []

    paradigm_name_map = {
        'supervised': 'SL',
        'medbooster': 'NB',
        'simclr': 'SimCLR',
        'vicreg': 'VICReg',
        'simim': 'SimMIM',
        'mae': 'MAE'
    }

    desired_order = ['supervised', 'medbooster', 'simclr', 'vicreg', 'simim', 'mae']
    path_dict = {extract_paradigm(Path(p)).lower(): p for p in paths}
    sorted_paths = [path_dict[k] for k in desired_order if k in path_dict]

    for path_str in sorted_paths:
        path = Path(path_str)
        mri_file = path / 'avg_image.npy'
        sal_file = path / 'avg_saliency.npy'
        if not (mri_file.exists() and sal_file.exists()):
            print(f"⚠️ Missing files in: {path}")
            continue

        avg_image = np.load(mri_file)
        avg_saliency = np.load(sal_file)
        avg_saliency = (avg_saliency - avg_saliency.min()) / (avg_saliency.ptp() + 1e-8)

        overlays.append((avg_image, avg_saliency))
        raw_paradigm = extract_paradigm(path).lower()
        paradigms.append(paradigm_name_map.get(raw_paradigm, raw_paradigm.upper()))

    if len(overlays) != 6:
        print(f"❌ Expected 6 overlays, got {len(overlays)}.")
        return

    # Layout: 2 paradigms per row, total 3 rows
    # fig = plt.figure(figsize=(3.3, 7.5))
    fig = plt.figure(figsize=(9, 7.5))

    total_rows = 3 
    total_height = 0.01    # space for each saliency+overlay pair
    row_height = 1.0 / total_rows

    paradigm_width = 0.5
    gap_between_paradigms = 0.05
    title_height = 0.02
    image_height = row_height - title_height
    pair_gap = 0.01

    for i in range(6):
        row = i // 2
        col = i % 2

        # X offset: if right column, add gap
        x0 = col * (paradigm_width + gap_between_paradigms)

        y0 = 1.0 - (row + 1) * row_height

        image, saliency = overlays[i]

        # Title
        ax_title = fig.add_axes([x0, y0 + image_height, paradigm_width, title_height])
        ax_title.axis('off')
        ax_title.text(0.5, -0.6, paradigms[i], fontsize=14, fontweight='bold',
                      ha='center', va='bottom', transform=ax_title.transAxes)

        img_width = (paradigm_width - pair_gap) / 2

        # Saliency
        ax_sal = fig.add_axes([x0, y0, img_width, image_height])
        ax_sal.imshow(saliency, cmap='hot')
        ax_sal.axis('off')

        # Overlay
        ax_overlay = fig.add_axes([x0 + img_width + pair_gap, y0, img_width, image_height])
        ax_overlay.imshow(image, cmap='gray')
        ax_overlay.imshow(saliency, cmap='hot', alpha=0.5)
        ax_overlay.axis('off')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"✅ Saved final layout with spacing between paradigms to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    input_paths = [
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_11_ADAMW_short/seed4/AGE/mae/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_11_ADAMW_short/seed0/AGE/simim/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long/seed4/AGE/supervised/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long/seed3/AGE/vicreg/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long/seed4/AGE/medbooster/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_07_08_LARS_long_simclr_deit/seed4/AGE/simclr/labels_percentage_100/fold_0/saliency_maps_average"
    ]

    output_file = "saliency_comparison.png"
    plot_multiple_overlays(input_paths, save_path=output_file)


if __name__ == '__main__':
    input_paths = [
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_11_ADAMW_short/seed4/AGE/mae/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_11_ADAMW_short/seed0/AGE/simim/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long/seed4/AGE/supervised/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long/seed3/AGE/vicreg/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long/seed4/AGE/medbooster/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_07_08_LARS_long_simclr_deit/seed4/AGE/simclr/labels_percentage_100/fold_0/saliency_maps_average"
    ]

    output_file = "saliency_comparison.png"
    plot_multiple_overlays(input_paths, save_path=output_file)

from PIL import Image

def resize_image(input_path, output_path, target_width_inch=3.3, dpi=300):
    img = Image.open(input_path)
    target_width_px = int(target_width_inch * dpi)
    aspect_ratio = img.height / img.width
    target_height_px = int(target_width_px * aspect_ratio)

    resized_img = img.resize((target_width_px, target_height_px), Image.LANCZOS)
    resized_img.save(output_path, dpi=(dpi, dpi))
    print(f"✅ Resized image saved to {output_path}")

# Example usage
resize_image("saliency_comparison.png", "saliency_comparison_resized.png")
