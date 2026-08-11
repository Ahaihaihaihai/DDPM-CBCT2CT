"""
Per-SLICE soft-tissue HU histograms for ONE patient (with region option)
------------------------------------------------------------------------
For a single patient, makes per-axial-slice histograms comparing CBCT original /
CBCT corrected / CT, plus a per-slice CSV.

REGION_MODE selects which region each histogram covers:
  "all"  -> all in-body voxels (body mask, minus out-of-FOV padding). Includes
            bone -> wide HU range.
  "soft" -> soft-tissue only (in-body AND CT within [SOFT_LO, SOFT_HI]).
  "both" -> both, side by side in one PNG (left = all in-body, right = soft).

The region is defined using the CT (trusted). No HU value is shifted/normalized.

Outputs (in your run directory):
  <OUT_DIR>/
      per_slice_stats.csv           <- one row per (slice, region)
      slices/slice_0042.png
      ...

NOTE: NOT run for you. Illustrative numbers only.
"""

import os
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ========================= CONFIG (edit these) =========================
ORIG_ROOT = "/data/3THDD/dataset/CBCT2CT/brain/"            # original NIfTI
CORR_ROOT = "/data/3THDD/dataset/CBCT2CT/corrected_brain/"  # corrected NIfTI
PATIENT   = "2BA028"           # single patient to inspect
CT_NAME   = "ct.nii.gz"
CBCT_NAME = "cbct.nii.gz"
MASK_NAME = "mask.nii.gz"
USE_PROVIDED_MASK = True

REGION_MODE = "both"           # "all", "soft", or "both"
SOFT_LO, SOFT_HI = 0.0, 80.0   # soft-tissue window (defined on CT)
CBCT_FLOOR = -900.0            # out-of-FOV floor (from original cbct)

# which slices: None = all. Otherwise a (start, stop) range. STEP skips slices.
SLICE_RANGE = (95, 105)
SLICE_STEP  = 1
MIN_VOXELS  = 200              # skip a region on a slice if it has fewer voxels

OUT_DIR = f"per_slice_{PATIENT}"
HIST_BIN_WIDTH = 2.0
ZOOM = True                    # auto-crop x-range to where the data actually is
# =======================================================================


def load_vol(path):
    return nib.load(path).get_fdata().astype(np.float32)


def stats(v):
    p1, p50, p99 = np.percentile(v, [1, 50, 99])
    return {"n": int(v.size), "mean": round(float(v.mean()), 2),
            "std": round(float(v.std()), 2),
            "p1": round(float(p1), 2), "p50": round(float(p50), 2),
            "p99": round(float(p99), 2)}


def _draw_panel(ax, label, ct_v, orig_v, corr_v, z):
    all_min = min(ct_v.min(), orig_v.min(), corr_v.min())
    all_max = max(ct_v.max(), orig_v.max(), corr_v.max())
    bins = np.arange(all_min, all_max + HIST_BIN_WIDTH, HIST_BIN_WIDTH)

    ax.hist(orig_v, bins=bins, alpha=0.45, density=True, label="CBCT original")
    ax.hist(corr_v, bins=bins, alpha=0.45, density=True, label="CBCT corrected")
    ax.hist(ct_v,   bins=bins, alpha=0.45, density=True, label="CT")

    if ZOOM:
        lo = min(np.percentile(v, 0.5)  for v in (ct_v, orig_v, corr_v))
        hi = max(np.percentile(v, 99.5) for v in (ct_v, orig_v, corr_v))
        pad = 0.05 * (hi - lo) if hi > lo else 10.0
        ax.set_xlim(lo - pad, hi + pad)

    ax.set_xlabel("HU")
    ax.set_ylabel("density")
    ax.set_title(f"{PATIENT} - slice {z} - {label}")
    ax.legend()


def save_slice_hist(panels, z, out_png):
    """panels: list of (label, ct_v, orig_v, corr_v)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 5), squeeze=False)
    for ax, (label, ct_v, orig_v, corr_v) in zip(axes[0], panels):
        _draw_panel(ax, label, ct_v, orig_v, corr_v, z)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ct   = load_vol(os.path.join(ORIG_ROOT, PATIENT, CT_NAME))
    orig = load_vol(os.path.join(ORIG_ROOT, PATIENT, CBCT_NAME))
    corr = load_vol(os.path.join(CORR_ROOT, PATIENT, CBCT_NAME))
    for name, v in (("orig", orig), ("corr", corr)):
        if v.shape != ct.shape:
            raise SystemExit(f"shape mismatch ({name}): {v.shape} vs CT {ct.shape}")

    mask_path = os.path.join(ORIG_ROOT, PATIENT, MASK_NAME)
    if USE_PROVIDED_MASK and os.path.exists(mask_path):
        body = load_vol(mask_path) > 0
    else:
        body = ct > -500

    # which region panels to produce, in display order
    wanted = []
    if REGION_MODE in ("all", "both"):
        wanted.append("all in-body")
    if REGION_MODE in ("soft", "both"):
        wanted.append("soft-tissue")
    if not wanted:
        raise SystemExit(f"REGION_MODE must be 'all', 'soft', or 'both' (got {REGION_MODE!r})")

    n_slices = ct.shape[2]
    start, stop = (0, n_slices) if SLICE_RANGE is None else SLICE_RANGE
    slice_ids = list(range(start, min(stop, n_slices), SLICE_STEP))

    slices_dir = os.path.join(OUT_DIR, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "per_slice_stats.csv")

    fields = ["slice", "region", "n_voxels",
              "ct_p50", "orig_p50", "corr_p50",
              "shift_orig_ct", "shift_corr_ct"]
    rows = []

    print(f"Patient {PATIENT}: {n_slices} slices total; processing {len(slice_ids)} "
          f"(regions: {', '.join(wanted)}).")

    for z in slice_ids:
        ct_s, o_s, c_s = ct[:, :, z], orig[:, :, z], corr[:, :, z]
        valid = o_s > CBCT_FLOOR                       # drop out-of-FOV padding
        region_all  = body[:, :, z] & valid
        region_soft = region_all & (ct_s > SOFT_LO) & (ct_s < SOFT_HI)
        region_of = {"all in-body": region_all, "soft-tissue": region_soft}

        panels = []
        for label in wanted:
            m = region_of[label]
            nv = int(m.sum())
            if nv < MIN_VOXELS:
                continue
            ct_v, o_v, c_v = ct_s[m], o_s[m], c_s[m]
            panels.append((label, nv, ct_v, o_v, c_v))

        if not panels:
            continue

        save_slice_hist([(l, ct_v, o_v, c_v) for (l, nv, ct_v, o_v, c_v) in panels],
                        z, os.path.join(slices_dir, f"slice_{z:04d}.png"))

        line = [f"slice {z:>4}:"]
        for (label, nv, ct_v, o_v, c_v) in panels:
            cs, os_, cs2 = stats(ct_v), stats(o_v), stats(c_v)
            shift_o = round(os_["mean"] - cs["mean"], 2)
            shift_c = round(cs2["mean"] - cs["mean"], 2)
            rows.append({"slice": z, "region": label, "n_voxels": nv,
                         "ct_p50": cs["p50"], "orig_p50": os_["p50"], "corr_p50": cs2["p50"],
                         "shift_orig_ct": shift_o, "shift_corr_ct": shift_c})
            line.append(f"[{label}: corr p50={cs2['p50']:>7}, shift={shift_c:+.1f}]")
        print("  " + " ".join(line))

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. {len(rows)} rows written.")
    print(f"CSV    -> {csv_path}")
    print(f"Slices -> {slices_dir}/slice_XXXX.png")


if __name__ == "__main__":
    main()