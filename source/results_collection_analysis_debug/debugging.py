import numpy as np
from pathlib import Path

# ✅ Sostituisci con i tuoi path
preds_path = Path("/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_16_LARS_long_no_augm/seed0/AGE/supervised/labels_percentage_100/fold_0/valdiation_outputs_and_targets/y_predicted_validation_epoch402.npy")
targets_path = Path("/Ironman/scratch/Andrea/med-booster/REVISION1/EXPERIMENTS_ABLATION_2025_06_16_LARS_long_no_augm/seed0/AGE/supervised/labels_percentage_100/fold_0/valdiation_outputs_and_targets/y_target_validation_epoch402.npy")

# ✅ Carica gli array
y_pred = np.load(preds_path)
y_true = np.load(targets_path)

# ✅ Controlli base
print("🔍 Controlli di integrità sugli array:")
print(f" - y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
print(f" - y_true shape: {y_true.shape}, dtype: {y_true.dtype}")

if y_pred.shape != y_true.shape:
    print("⚠️ Le shape non corrispondono!")
else:
    print("✅ Le shape corrispondono")

# ✅ Statistiche
print("\n📊 Statistiche:")
print(f" - y_pred: mean={y_pred.mean():.4f}, std={y_pred.std():.4f}, min={y_pred.min():.4f}, max={y_pred.max():.4f}")
print(f" - y_true: mean={y_true.mean():.4f}, std={y_true.std():.4f}, min={y_true.min():.4f}, max={y_true.max():.4f}")

# ✅ Primi e ultimi valori
print("\n🧪 Primi 10 valori:")
print(" - y_pred[:10]:", y_pred[:10])
print(" - y_true[:10]:", y_true[:10])

print("\n🧪 Ultimi 10 valori:")
print(" - y_pred[-10:]:", y_pred[-10:])
print(" - y_true[-10:]:", y_true[-10:])
