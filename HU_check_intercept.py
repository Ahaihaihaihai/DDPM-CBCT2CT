# -*- coding: utf-8 -*-
# Diagnostic: compare CBCT HU statistics BEFORE vs AFTER forcing the intercept to -1024.
# Shows WHY the calibration correction is needed. With the original (inconsistent) tag,
# some series are shifted relative to CT (e.g. 2BA intercept=0 -> CBCT reads ~+1024 HU
# too high); forcing -1024 realigns CBCT with CT for every series.
#
# CLEAN version: reads RAW HU straight from DICOM (NO clipping to [-1000, 2000]), so the
# offset is measured faithfully. Images are only resized to 256x256 (to align CBCT / CT /
# mask grids, which can differ in native size) -- no windowing, no normalization.
# Compared inside the body mask (ROI), vs GT CT. Does NOT need a checkpoint.
#
# For each series you should see:
#   2BA (orig intercept 0)     -> large, CONSISTENT BEFORE shift (~+1024 HU) -> ~0 AFTER
#   2BB (orig intercept -1000) -> small BEFORE shift (~+24 HU vs AFTER)
#   2BC (orig intercept -1024) -> already correct                -> before == after
import os
import numpy as np
import torch
import torch.nn.functional as F
import pydicom
from datasets import BrainDataset

# ===================== CONFIG =====================
dataset_name = "/data/3THDD/dataset/CBCT2CT/brain_DICOM/"
N_SLICES = 12
TARGET_SIZE = (256, 256)
FORCED_INTERCEPT = -1024.0
# ==================================================


def resize_like_pipeline(arr, mode):
    """Resize a raw 2D array to TARGET_SIZE (align grids) WITHOUT clipping/normalizing."""
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    if mode == "bilinear":
        t = F.interpolate(t, size=TARGET_SIZE, mode="bilinear", align_corners=False)
    else:
        t = F.interpolate(t, size=TARGET_SIZE, mode="nearest")
    return t.squeeze().numpy()


# Build the dataset just to get the (cbct, ct, mask) file paths and the split.
ds = BrainDataset(root=dataset_name, mode="test", target_size=TARGET_SIZE,
                  apply_mask_to_image=True)
idxs = np.linspace(0, len(ds.slices) - 1, N_SLICES).astype(int)

print(f"\nCBCT HU inside the body mask (ROI) vs GT CT  |  RAW HU, no clipping  |  "
      f"BEFORE = original tag, AFTER = forced {FORCED_INTERCEPT:.0f}")
print(f"{'patient/slice':20s} {'orig_int':>9s} "
      f"{'Δbefore':>9s} {'MAEbefore':>10s}   {'Δafter':>9s} {'MAEafter':>9s}")
print("-" * 76)

rows = []
for idx in idxs:
    idx = int(idx)
    cbct_f, ct_f, mask_f = ds.slices[idx]

    # RAW HU (no clip). CBCT loaded twice: original tag vs forced -1024. CT uses its own tag.
    cbct_before = resize_like_pipeline(
        BrainDataset._load_dicom_array(cbct_f, force_intercept=None), "bilinear")
    cbct_after  = resize_like_pipeline(
        BrainDataset._load_dicom_array(cbct_f, force_intercept=FORCED_INTERCEPT), "bilinear")
    gt          = resize_like_pipeline(
        BrainDataset._load_dicom_array(ct_f), "bilinear")
    mask        = resize_like_pipeline(
        BrainDataset._load_dicom_mask(mask_f), "nearest") > 0.5

    if mask.sum() == 0:
        continue

    b, a, g = cbct_before[mask], cbct_after[mask], gt[mask]
    shift_b = float(np.mean(b - g)); mae_b = float(np.mean(np.abs(b - g)))
    shift_a = float(np.mean(a - g)); mae_a = float(np.mean(np.abs(a - g)))

    orig_int = float(getattr(pydicom.dcmread(cbct_f, stop_before_pixels=True),
                             "RescaleIntercept", 0.0))

    # patient id & slice name from path: .../<patient>/cbct/<slice>.dcm
    patient    = os.path.basename(os.path.dirname(os.path.dirname(cbct_f)))
    slice_name = os.path.splitext(os.path.basename(cbct_f))[0]
    rows.append((shift_b, mae_b, shift_a, mae_a))
    print(f"{patient+'/'+slice_name:20s} {orig_int:>9.0f} "
          f"{shift_b:>9.0f} {mae_b:>10.0f}   {shift_a:>9.0f} {mae_a:>9.0f}")

print("-" * 76)
if rows:
    sb_, mb_, sa_, ma_ = np.mean(rows, axis=0)
    print(f"{'AVERAGE':20s} {'':>9s} {sb_:>9.0f} {mb_:>10.0f}   {sa_:>9.0f} {ma_:>9.0f}")

print("\nHow to read:")
print(" - Δbefore/Δafter = mean(CBCT - GT) inside the mask (systematic HU offset).")
print(" - A large Δbefore that drops to ~0 in Δafter = the forced -1024 fixed a real")
print("   calibration shift (that series' CBCT was on the wrong HU scale).")
print(" - No clipping here, so 2BA shows a clean, consistent ~+1024 before-shift and no")
print("   negative artifacts. Series already at -1024 (2BC) show before == after.")
