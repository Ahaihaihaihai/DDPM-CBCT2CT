# -*- coding: utf-8 -*-
# Diagnostic: check HU statistics (ROI) of CBCT vs GT on a few test slices.
# Does NOT need a checkpoint. The goal is to see whether there is an HU
# calibration shift or an abnormally high baseline MAE.
import numpy as np
import torch
from datasets import BrainDataset

# ===================== CONFIG =====================
dataset_name = "/data/3THDD/dataset/CBCT2CT/brain_DICOM/"
N_SLICES = 12
HU_MIN, HU_MAX = -1000, 2000
# ==================================================


def to_hu(a):
    return (a + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN


ds = BrainDataset(root=dataset_name, mode="test", target_size=(256, 256),
                  apply_mask_to_image=True)
idxs = np.linspace(0, len(ds) - 1, N_SLICES).astype(int)

print(f"\n{'patient/slice':22s} {'CBCT_roi mean[min,max]':30s} {'GT_roi mean[min,max]':30s} "
      f"{'meanΔ':>7s} {'MAE':>7s}")
print("-" * 105)

mae_all, shift_all = [], []
for idx in idxs:
    s = ds[int(idx)]
    cbct = to_hu(s["CBCT"]).squeeze().numpy()
    gt   = to_hu(s["pCT"]).squeeze().numpy()
    m = s["mask"].squeeze().numpy() > 0
    if m.sum() == 0:
        continue

    c, g = cbct[m], gt[m]
    mae = float(np.mean(np.abs(c - g)))
    shift = float(np.mean(c - g))          # systematic offset (CBCT - GT)
    mae_all.append(mae); shift_all.append(shift)

    pid = s.get("patient", "?"); sl = s.get("slice_name", str(idx))
    print(f"{pid+'/'+sl:22s} "
          f"{f'{c.mean():6.0f} [{c.min():5.0f},{c.max():5.0f}]':30s} "
          f"{f'{g.mean():6.0f} [{g.min():5.0f},{g.max():5.0f}]':30s} "
          f"{shift:7.0f} {mae:7.0f}")

print("-" * 105)
print(f"Average: meanΔ (CBCT-GT) = {np.mean(shift_all):.1f} HU | MAE = {np.mean(mae_all):.1f} HU")
print("\nHow to read:")
print(" - If meanΔ is large (e.g. hundreds of HU) & consistent -> there is an HU calibration OFFSET (CBCT is not true HU).")
print(" - If CBCT and GT means are similar but MAE is still high -> bad registration/artifacts, not an offset.")
print(" - Compare with the compare_tstart baseline (517 HU). If it is also ~500 here -> the data really is like that;")
print("   if it is much smaller here -> there is a difference in the metric pipeline that needs investigating.")
