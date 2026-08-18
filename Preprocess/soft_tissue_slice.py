"""
Per-SLICE HU histograms for ONE patient (region + stream options)
-----------------------------------------------------------------
Per-axial-slice histograms comparing CBCT original / corrected / CT for a single
patient, plus a per-slice CSV. Toggles match compare_configurable_v2.py:

  REGION_MODE : "all" | "soft" | "both"
  STREAMS     : subset of "ct","orig","corr"  (which curves to compute/plot)

Region is defined on the CT. No HU value is shifted/normalized -- read as stored.

Outputs (under <OUT_DIR>/):
  per_slice_stats.csv               <- one row per (slice, region, stream)
  slices/slice_0042.png

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
ORIG_ROOT = "/data/3THDD/dataset/CBCT2CT/brain/"
CORR_ROOT = "/data/3THDD/dataset/CBCT2CT/corrected_brain/"
PATIENT   = "2BA028"           # single patient to inspect
CT_NAME   = "ct.nii.gz"
CBCT_NAME = "cbct.nii.gz"
MASK_NAME = "mask.nii.gz"
USE_PROVIDED_MASK = True

# ---- toggles ----
REGION_MODE = "both"           # "all" | "soft" | "both"
STREAMS = ["orig", "corr", "ct"]   # subset of "ct","orig","corr"

SOFT_LO, SOFT_HI = 0.0, 80.0
CBCT_FLOOR = -900.0

SLICE_RANGE = (95, 105)        # None = all slices; else (start, stop)
SLICE_STEP  = 1
MIN_VOXELS  = 200

OUT_DIR = f"per_slice_{PATIENT}"
HIST_BIN_WIDTH = 2.0
ZOOM = True
# =======================================================================

STREAM_LABEL = {"ct": "CT", "orig": "CBCT original", "corr": "CBCT corrected"}
STREAM_COLOR = {"orig": "C0", "corr": "C1", "ct": "C2"}
STREAM_ORDER = ["orig", "corr", "ct"]


def load_vol(path):
    return nib.load(path).get_fdata().astype(np.float32)


def stats(v):
    p1, p50, p99 = np.percentile(v, [1, 50, 99])
    return {"n": int(v.size), "mean": round(float(v.mean()), 2),
            "std": round(float(v.std()), 2),
            "p1": round(float(p1), 2), "p50": round(float(p50), 2),
            "p99": round(float(p99), 2)}


def draw_panel(ax, label, valdict):
    order = [s for s in STREAM_ORDER if s in valdict]
    all_min = min(valdict[s].min() for s in order)
    all_max = max(valdict[s].max() for s in order)
    bins = np.arange(all_min, all_max + HIST_BIN_WIDTH, HIST_BIN_WIDTH)
    for s in order:
        ax.hist(valdict[s], bins=bins, alpha=0.45, density=True,
                label=STREAM_LABEL[s], color=STREAM_COLOR[s])
    if ZOOM:
        lo = min(np.percentile(valdict[s], 0.5)  for s in order)
        hi = max(np.percentile(valdict[s], 99.5) for s in order)
        pad = 0.05 * (hi - lo) if hi > lo else 10.0
        ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("HU"); ax.set_ylabel("density"); ax.set_title(label); ax.legend()


def save_slice_hist(panels, out_png):
    """panels: list of (label, valdict)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 5), squeeze=False)
    for ax, (label, valdict) in zip(axes[0], panels):
        draw_panel(ax, label, valdict)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    bad = [s for s in STREAMS if s not in STREAM_LABEL]
    if not STREAMS or bad:
        raise SystemExit(f"STREAMS must be a subset of ['ct','orig','corr'] (bad: {bad})")
    sel = [s for s in STREAM_ORDER if s in STREAMS]
    load_corr = "corr" in sel

    ct   = load_vol(os.path.join(ORIG_ROOT, PATIENT, CT_NAME))
    orig = load_vol(os.path.join(ORIG_ROOT, PATIENT, CBCT_NAME))
    corr = load_vol(os.path.join(CORR_ROOT, PATIENT, CBCT_NAME)) if load_corr else None
    for name, v in (("orig", orig), ("corr", corr)):
        if v is not None and v.shape != ct.shape:
            raise SystemExit(f"shape mismatch ({name}): {v.shape} vs CT {ct.shape}")

    mask_path = os.path.join(ORIG_ROOT, PATIENT, MASK_NAME)
    body = (load_vol(mask_path) > 0) if (USE_PROVIDED_MASK and os.path.exists(mask_path)) else (ct > -500)

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

    fields = ["slice", "region", "stream", "n_voxels", "mean", "std",
              "p1", "p50", "p99", "shift_vs_ct"]
    rows = []

    print(f"Patient {PATIENT}: {n_slices} slices total; processing {len(slice_ids)} "
          f"(regions: {', '.join(wanted)}; streams: {sel}).")

    for z in slice_ids:
        ct_s = ct[:, :, z]
        o_s = orig[:, :, z]
        c_s = corr[:, :, z] if load_corr else None
        valid = o_s > CBCT_FLOOR
        region_all  = body[:, :, z] & valid
        region_soft = region_all & (ct_s > SOFT_LO) & (ct_s < SOFT_HI)
        region_of = {"all in-body": region_all, "soft-tissue": region_soft}

        panels = []
        line = [f"slice {z:>4}:"]
        for label in wanted:
            m = region_of[label]
            nv = int(m.sum())
            if nv < MIN_VOXELS:
                continue
            ct_ref = ct_s[m]                      # always available for shift
            valdict = {}
            if "ct" in sel:
                valdict["ct"] = ct_ref
            if "orig" in sel:
                valdict["orig"] = o_s[m]
            if "corr" in sel:
                valdict["corr"] = c_s[m]

            ct_mean = float(ct_ref.mean())
            for s in [x for x in STREAM_ORDER if x in valdict]:
                st = stats(valdict[s])
                shift = "" if s == "ct" else round(st["mean"] - ct_mean, 2)
                rows.append({"slice": z, "region": label, "stream": s,
                             "n_voxels": st["n"], "mean": st["mean"], "std": st["std"],
                             "p1": st["p1"], "p50": st["p50"], "p99": st["p99"],
                             "shift_vs_ct": shift})
            panels.append((label, valdict))
            line.append(f"[{label}: n={nv}]")

        if not panels:
            continue
        save_slice_hist(panels, os.path.join(slices_dir, f"slice_{z:04d}.png"))
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