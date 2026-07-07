# -*- coding: utf-8 -*-
# DICOM-level diagnostic: check RescaleSlope/Intercept & raw-vs-HU value range,
# PLUS geometry (PixelSpacing, grid size, physical FOV) to diagnose shape
# distortion (oval -> round) caused by resizing to a square grid.
import os
import glob
import numpy as np
import pydicom

# ===================== CONFIG =====================
root = "/data/3THDD/dataset/CBCT2CT/brain_DICOM/"
PATIENTS = ["2BA007", "2BB011", "2BC014"]   # one representative per series (adjust if needed)
SUBS = ["cbct", "ct"]
# ==================================================

for pid in PATIENTS:
    print("=" * 70)
    print(f"PATIENT {pid}")
    for sub in SUBS:
        files = sorted(glob.glob(os.path.join(root, pid, sub, "*.dcm")))
        if not files:
            print(f"  [{sub}] no .dcm files")
            continue
        f = files[len(files) // 2]   # middle slice
        d = pydicom.dcmread(f)
        raw = d.pixel_array.astype(np.float32)
        slope = float(getattr(d, "RescaleSlope", 1.0))
        inter = float(getattr(d, "RescaleIntercept", 0.0))
        hu = raw * slope + inter

        modality = getattr(d, "Modality", "?")
        bits = getattr(d, "BitsStored", "?")
        signed = getattr(d, "PixelRepresentation", "?")  # 0=unsigned, 1=signed

        # --- geometry ---
        rows = int(getattr(d, "Rows", raw.shape[0]))
        cols = int(getattr(d, "Columns", raw.shape[1]))
        ps = d.get("PixelSpacing", None)            # [row(mm), col(mm)]
        thick = getattr(d, "SliceThickness", None)

        if ps is not None:
            row_sp = float(ps[0])
            col_sp = float(ps[1])
            fov_h = rows * row_sp   # physical height (mm)
            fov_w = cols * col_sp   # physical width (mm)
            ps_str = f"[{row_sp:.4f}, {col_sp:.4f}] mm"
            fov_str = f"{fov_w:.1f}(W) x {fov_h:.1f}(H) mm  ratio W/H={fov_w/fov_h:.3f}"
        else:
            ps_str = "NONE"
            fov_str = "cannot compute (PixelSpacing empty)"

        print(f"  [{sub}] {os.path.basename(f)}")
        print(f"        Modality={modality}  Slope={slope}  Intercept={inter}  "
              f"BitsStored={bits}  PixelRep={signed}")
        print(f"        GRID: {rows}(rows) x {cols}(cols) px   "
              f"ratio px H/W={rows/cols:.3f}")
        print(f"        PixelSpacing={ps_str}   SliceThickness={thick}")
        print(f"        Physical FOV: {fov_str}")
        print(f"        RAW : min={raw.min():7.0f}  max={raw.max():7.0f}  mean={raw.mean():7.0f}")
        print(f"        HU  : min={hu.min():7.0f}  max={hu.max():7.0f}  mean={hu.mean():7.0f}")

print("=" * 70)
print("\nWhat to look for (geometry / oval vs round shape):")
print(" - Compare the physical FOV 'ratio W/H' of CBCT & CT. If it is != 1.0 (e.g. ~0.81),")
print("   the original shape really is oval. Once resized to 256x256 (ratio 1.0),")
print("   the oval becomes round -> that is the source of 'sCT looking round', not the model.")
print(" - If PixelSpacing is anisotropic (row_sp != col_sp), that also makes")
print("   the shape look different between the pixel grid and the physical display.")
print("\nWhat to look for (HU calibration, as before):")
print(" - Compare CBCT vs CT Intercept in each series (2BA intercept=0 vs CT -1024).")
print(" - Compare RAW mean over similar tissue across series.")
print(" - A different PixelRep (signed/unsigned) can cause a shift.")
