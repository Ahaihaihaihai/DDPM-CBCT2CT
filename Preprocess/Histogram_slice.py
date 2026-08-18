# Basic per-slice histogram: CT vs CBCT soft tissue, ONE patient
# Simple version -- plain plt.hist, one PNG per slice. NOT run for you.

import os
import nibabel as nib
import matplotlib
matplotlib.use("Agg")            # save to file (headless server, no pop-up window)
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BASE_DIR = "/data/3THDD/dataset/CBCT2CT/brain"   # folder with <patient>/ct.nii.gz, cbct.nii.gz
PATIENT = "2BA028"               # one patient (per-slice is for a single patient)
SOFT_LO, SOFT_HI = 0, 80         # soft-tissue HU window (defined on CT)
BINS = 100                       # number of histogram bars
SLICE_START = 95                 # first slice to plot
SLICE_END = 105                  # last slice to plot (exclusive)
OUT_DIR = "hist_per_slice"       # where the PNGs go
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# load the two images once for this patient
ct = nib.load(os.path.join(BASE_DIR, PATIENT, "ct.nii.gz")).get_fdata()
cbct = nib.load(os.path.join(BASE_DIR, PATIENT, "cbct.nii.gz")).get_fdata()

print(PATIENT, "has", ct.shape[2], "slices")

# loop over the chosen slice range
for z in range(SLICE_START, SLICE_END):
    # 1. take one slice from each image
    ct_slice = ct[:, :, z]
    cbct_slice = cbct[:, :, z]

    # 2. simple soft-tissue mask from the CT slice
    mask = (ct_slice > SOFT_LO) & (ct_slice < SOFT_HI)

    # 3. values inside the mask
    ct_values = ct_slice[mask]
    cbct_values = cbct_slice[mask]

    # skip a slice that has no soft tissue (empty mask)
    if ct_values.size == 0:
        print("  slice", z, "empty, skipped")
        continue

    # 4. basic histogram
    plt.figure()
    plt.hist(ct_values,   bins=BINS, alpha=0.5, label="CT")
    plt.hist(cbct_values, bins=BINS, alpha=0.5, label="CBCT")
    plt.xlabel("HU")
    plt.ylabel("number of voxels")
    plt.title(PATIENT + " - slice " + str(z))
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "slice_" + str(z) + ".png"))
    plt.close()

    print("  saved slice", z)

print("done. PNGs are in", OUT_DIR)