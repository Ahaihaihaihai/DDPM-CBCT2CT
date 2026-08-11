# -*- coding: utf-8 -*-
# Render the .npy output of Test_condition.py -> PNG. Can be re-run anytime, any window,
# without re-sampling. Used when Test_condition is run with SAVE_PNG=False.
#
# Input structure  : <npy_root>/<patient>/sCT/<slice>.npy  = (3,256,256) HU [CBCT, sCT, GT]
# Output structure : <img_root>/<patient>/<slice>.png      = panel [CBCT | sCT | GT]
#                    (+ <slice>_sct.png if SAVE_SCT_ONLY=True)
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
out_name      = "test_1"  # output folder name under ./test/
npy_root      = f"./test/{out_name}/npy"
img_root      = f"./test/{out_name}/images"
WIN_LO, WIN_HI = -500, 1500    # display window HU (bone preset: center 500 / width 2000 from npy_to_dicom.py)
fmt           = "png"          # "png" or "jpg"
SAVE_PANEL    = True           # panel [CBCT | sCT | GT]
SAVE_SCT_ONLY = False          # also save sCT by itself (without the panel)
OVERWRITE     = False          # False = skip files that already exist
# ==================================================


def window(img):
    return np.clip(img, WIN_LO, WIN_HI)


# input: <npy_root>/<patient>/sCT/<slice>.npy
files = sorted(glob.glob(os.path.join(npy_root, "*", "sCT", "*.npy")))
if not files:
    print(f"No .npy in '{npy_root}/<patient>/sCT/'. Check out_name / path.")
    print("Hint: structure from Test_condition.py = npy/<patient>/sCT/<slice>.npy")
    raise SystemExit

print(f"Found {len(files)} slices. Output -> {img_root}/<patient>/  (window [{WIN_LO},{WIN_HI}])")
made = skipped = 0

for f in files:
    # f = .../npy/<patient>/sCT/<slice>.npy
    slice_name = os.path.splitext(os.path.basename(f))[0]
    patient    = os.path.basename(os.path.dirname(os.path.dirname(f)))

    out_pat    = os.path.join(img_root, patient)
    panel_path = os.path.join(out_pat, f"{slice_name}.{fmt}")
    sct_path   = os.path.join(out_pat, f"{slice_name}_sct.{fmt}")

    want_panel = SAVE_PANEL    and (OVERWRITE or not os.path.exists(panel_path))
    want_sct   = SAVE_SCT_ONLY and (OVERWRITE or not os.path.exists(sct_path))
    if not (want_panel or want_sct):
        skipped += 1
        continue

    os.makedirs(out_pat, exist_ok=True)
    trio = np.load(f)                         # (3,256,256) = [CBCT, sCT, GT] HU
    cbct, sct, gt = trio[0], trio[1], trio[2]

    if want_panel:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        for ax, im, t in zip(axes, [cbct, sct, gt], ["CBCT", "sCT (paper)", "GT (CT)"]):
            ax.imshow(window(im), cmap="gray", vmin=WIN_LO, vmax=WIN_HI)
            ax.set_title(t, fontsize=9); ax.axis("off")
        fig.suptitle(f"{patient} / {slice_name}", fontsize=10)
        fig.tight_layout()
        fig.savefig(panel_path, dpi=110, bbox_inches="tight")
        plt.close(fig)

    if want_sct:
        plt.imsave(sct_path, window(sct), cmap="gray", vmin=WIN_LO, vmax=WIN_HI)

    made += 1

print(f"Done. Made: {made} | Skipped: {skipped}")
