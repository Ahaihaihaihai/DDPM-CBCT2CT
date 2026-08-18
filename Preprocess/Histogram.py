# Basic per-patient histogram: CT vs CBCT soft tissue
# Simple version -- plain plt.hist, one PNG per patient. NOT run for you.

import os
import glob
import nibabel as nib
import matplotlib
matplotlib.use("Agg")            # save to file (headless server, no pop-up window)
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BASE_DIR = "/data/3THDD/dataset/CBCT2CT/brain"   # folder with <patient>/ct.nii.gz, cbct.nii.gz
PREFIX = "2BB"                    # which center to loop over
SOFT_LO, SOFT_HI = 0, 80         # soft-tissue HU window (defined on CT)
BINS = 100                       # number of histogram bars
OUT_DIR = "hist_per_patient"     # where the PNGs go
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# find every patient folder that starts with PREFIX (loop from soft_tissue_analysis.py)
ct_paths = sorted(glob.glob(os.path.join(BASE_DIR, PREFIX + "*", "ct.nii.gz")))
print("found", len(ct_paths), "patients")

for ct_path in ct_paths:
    folder = os.path.dirname(ct_path)
    pid = os.path.basename(folder)

    # 1. load the two images as HU arrays
    ct = nib.load(ct_path).get_fdata()
    cbct = nib.load(os.path.join(folder, "cbct.nii.gz")).get_fdata()

    # 2. make a simple soft-tissue mask FROM the CT (True where CT is soft tissue)
    mask = (ct > SOFT_LO) & (ct < SOFT_HI)

    # 3. take the HU values inside that mask, for both images
    ct_values = ct
    cbct_values = cbct

    # 4. draw a basic histogram and save it
    plt.figure()
    plt.hist(ct_values,   bins=BINS, alpha=0.5, label="CT")
    plt.hist(cbct_values, bins=BINS, alpha=0.5, label="CBCT")
    plt.xlabel("HU")
    plt.ylabel("number of voxels")
    plt.title(pid)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, pid + ".png"))
    plt.close()                  # close so figures don't pile up in memory

    print("  saved", pid)

print("done. PNGs are in", OUT_DIR)s