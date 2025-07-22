import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim  # ✅ NEW

# ✅ MUST be defined before using
def extract_paradigm(path: Path) -> str:
    parts = path.parts
    if 'AGE' in parts:
        idx = parts.index('AGE')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "Unknown"


def plot_multiple_overlays(paths, save_path=None, mode="difference"):
    overlays = []
    paradigms = []

    paradigm_name_map = {
        'supervised': 'SL',
        'medbooster': 'NB',
        'simclr': 'SimCLR',
        'vicreg': 'VICReg',
        'simim': 'SimMIM',
        'mae': 'MaskedAE'
    }

    desired_order = ['supervised', 'medbooster', 'simclr', 'vicreg', 'simim', 'mae']
    path_dict = {extract_paradigm(Path(p)).lower(): p for p in paths}
    sorted_paths = [path_dict[k] for k in desired_order if k in path_dict]

    sl_saliency = None
    sl_index = None
    saliencies = []

    for i, path_str in enumerate(sorted_paths):
        path = Path(path_str)
        mri_file = path / 'avg_image.npy'
        sal_file = path / 'avg_saliency.npy'
        if not (mri_file.exists() and sal_file.exists()):
            print(f"⚠️ Missing files in: {path}")
            continue

        avg_image = np.load(mri_file)
        avg_saliency = np.load(sal_file)

        # Normalize saliency between 0 and 1 for both display and comparison
        norm_saliency = (avg_saliency - avg_saliency.min()) / (avg_saliency.ptp() + 1e-8)
        overlays.append((avg_image, norm_saliency))
        saliencies.append(norm_saliency)

        raw_paradigm = extract_paradigm(path).lower()
        display_name = paradigm_name_map.get(raw_paradigm, raw_paradigm.upper())
        paradigms.append(display_name)

        if raw_paradigm == "supervised":
            sl_saliency = norm_saliency
            sl_index = i

    if len(overlays) != 6:
        print(f"❌ Expected 6 overlays, got {len(overlays)}.")
        return

    # Compute metric vs SL
    if sl_saliency is not None:
        for i in range(len(paradigms)):
            if i == sl_index:
                continue
            if mode == "difference":
                value = np.mean(np.abs(saliencies[i] - sl_saliency))
                paradigms[i] += f" (Δ: {value:.3f})"
            elif mode == "similarity":
                value = ssim(saliencies[i], sl_saliency, data_range=1.0)
                paradigms[i] += f" (SSIM: {value:.3f})"

    # Layout: 2 paradigms per row, total 3 rows
    fig = plt.figure(figsize=(9, 7.5))

    total_rows = 3 
    row_height = 1.0 / total_rows
    paradigm_width = 0.5
    gap_between_paradigms = 0.05
    title_height = 0.02
    image_height = row_height - title_height
    pair_gap = 0.01

    for i in range(6):
        row = i // 2
        col = i % 2
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
        print(f"✅ Saved final layout with {mode} to {save_path}")
    else:
        plt.show()


def resize_image(input_path, output_path, target_width_inch=3.3, dpi=300):
    img = Image.open(input_path)
    target_width_px = int(target_width_inch * dpi)
    aspect_ratio = img.height / img.width
    target_height_px = int(target_width_px * aspect_ratio)

    resized_img = img.resize((target_width_px, target_height_px), Image.LANCZOS)
    resized_img.save(output_path, dpi=(dpi, dpi))
    print(f"✅ Resized image saved to {output_path}")


if __name__ == '__main__':
    input_paths = [
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_11_ADAMW_short/seed4/AGE/mae/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_11_ADAMW_short/seed0/AGE/simim/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long/seed4/AGE/supervised/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long/seed3/AGE/vicreg/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_05_26_LARS_long/seed4/AGE/medbooster/labels_percentage_100/fold_0/saliency_maps_average",
        "/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_07_08_LARS_long_simclr_deit/seed4/AGE/simclr/labels_percentage_100/fold_0/saliency_maps_average"
    ]

    # Difference
    diff_file = "saliency_comparison_with_difference.png"
    plot_multiple_overlays(input_paths, save_path=diff_file, mode="difference")
    resize_image(diff_file, "saliency_comparison_with_difference_resized.png")

    # Similarity (SSIM)
    sim_file = "saliency_comparison_with_similarity.png"
    plot_multiple_overlays(input_paths, save_path=sim_file, mode="similarity")
    resize_image(sim_file, "saliency_comparison_with_similarity_resized.png")
