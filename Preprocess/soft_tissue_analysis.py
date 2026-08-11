"""
CBCT original vs corrected vs CT -- configurable comparison
-----------------------------------------------------------
Two toggles control everything:

  OUTPUT_MODE : "per_patient" -> one histogram PNG per patient
                "pooled"      -> one histogram from ALL patients combined
                "both"        -> produce both

  REGION_MODE : "soft" -> soft-tissue only (in-body AND CT in [SOFT_LO, SOFT_HI])
                "all"  -> full in-body area (body mask, minus out-of-FOV padding)
                "both" -> both, shown side by side in each PNG

The region is defined on the CT. No HU value is shifted/normalized -- read as
stored. Pooling sums every voxel into a 1-HU-bin accumulator (low memory).

Outputs (in your run directory, under <OUT_DIR>/):
  compare_stats.csv                 <- rows per (patient, region) + (ALL, region)
  pooled.png                        <- if OUTPUT_MODE includes pooled
  histograms/<patient>.png          <- if OUTPUT_MODE includes per_patient

NOTE: NOT run for you. Illustrative numbers only.
"""

import os
import csv
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ========================= CONFIG (edit these) =========================
ORIG_ROOT = "/data/3THDD/dataset/CBCT2CT/brain"            # original NIfTI root
CORR_ROOT = "/data/3THDD/dataset/CBCT2CT/corrected_brain"  # corrected NIfTI root
PREFIX = "2BB"                  # which center: "2BB", "2BC", "2BA"
CT_NAME = "ct.nii.gz"
CBCT_NAME = "cbct.nii.gz"
MASK_NAME = "mask.nii.gz"
USE_PROVIDED_MASK = True

# ---- the two main toggles ----
OUTPUT_MODE = "both"           # "per_patient" | "pooled" | "both"
REGION_MODE = "soft"           # "soft" | "all" | "both"

SOFT_LO, SOFT_HI = 0.0, 80.0   # soft-tissue window (defined on CT)
CBCT_FLOOR = -900.0            # drop CBCT out-of-FOV padding below this

OUT_DIR = "compare_2BB"
HIST_BIN_WIDTH = 2.0           # per-patient PNG bin width
HIST_LO, HIST_HI = -1024, 3000 # pooled accumulator support (1-HU bins)
# =======================================================================

SIZE = HIST_HI - HIST_LO + 1
BINS = np.arange(HIST_LO, HIST_HI + 1)


def load_vol(path):
    return nib.load(path).get_fdata().astype(np.float32)


def stats(v):
    p1, p50, p99 = np.percentile(v, [1, 50, 99])
    return {"n": int(v.size), "mean": round(float(v.mean()), 2),
            "std": round(float(v.std()), 2),
            "p1": round(float(p1), 2), "p50": round(float(p50), 2),
            "p99": round(float(p99), 2)}


def add_to_hist(hist, vals):
    if vals.size:
        hist += np.bincount(np.clip(np.rint(vals).astype(np.int64) - HIST_LO, 0, SIZE - 1),
                            minlength=SIZE)


def hist_stats_acc(hist):
    h = hist.astype(np.float64); n = h.sum()
    if n == 0:
        return None
    c = BINS.astype(np.float64)
    mean = float((c * h).sum() / n)
    std = float(np.sqrt(((c - mean) ** 2 * h).sum() / n))
    cum = np.cumsum(h)
    def pct(p):
        i = int(np.searchsorted(cum, p / 100.0 * n)); return float(c[min(i, len(c) - 1)])
    return {"n": int(n), "mean": round(mean, 2), "std": round(std, 2),
            "p1": pct(1), "p50": pct(50), "p99": pct(99)}


def _zoom(streams_stats):
    present = [s for s in streams_stats if s]
    lo = min(s["p1"] for s in present); hi = max(s["p99"] for s in present)
    pad = 0.05 * (hi - lo) if hi > lo else 10.0
    return lo - pad, hi + pad


def draw_panel_values(ax, label, ct_v, orig_v, corr_v):
    lo = min(ct_v.min(), orig_v.min(), corr_v.min())
    hi = max(ct_v.max(), orig_v.max(), corr_v.max())
    bins = np.arange(lo, hi + HIST_BIN_WIDTH, HIST_BIN_WIDTH)
    ax.hist(orig_v, bins=bins, alpha=0.45, density=True, label="CBCT original")
    ax.hist(corr_v, bins=bins, alpha=0.45, density=True, label="CBCT corrected")
    ax.hist(ct_v,   bins=bins, alpha=0.45, density=True, label="CT")
    x0, x1 = _zoom([stats(ct_v), stats(orig_v), stats(corr_v)])
    ax.set_xlim(x0, x1)
    ax.set_xlabel("HU"); ax.set_ylabel("density"); ax.set_title(label); ax.legend()


def draw_panel_acc(ax, label, acc_ct, acc_orig, acc_corr):
    x0, x1 = _zoom([hist_stats_acc(acc_ct), hist_stats_acc(acc_orig), hist_stats_acc(acc_corr)])
    sel = (BINS >= x0) & (BINS <= x1)
    for h, lab in ((acc_orig, "CBCT original"), (acc_corr, "CBCT corrected"), (acc_ct, "CT")):
        if h.sum():
            ax.plot(BINS[sel], h[sel].astype(np.float64) / h.sum(), label=lab, linewidth=1.6)
    ax.set_xlabel("HU"); ax.set_ylabel("normalized freq"); ax.set_title(label)
    ax.grid(alpha=0.3); ax.legend()


def region_masks(ct, cbct_o, body):
    valid = cbct_o > CBCT_FLOOR
    region_all = body & valid
    region_soft = region_all & (ct > SOFT_LO) & (ct < SOFT_HI)
    return {"all in-body": region_all, "soft-tissue": region_soft}


def wanted_regions():
    w = []
    if REGION_MODE in ("all", "both"):
        w.append("all in-body")
    if REGION_MODE in ("soft", "both"):
        w.append("soft-tissue")
    if not w:
        raise SystemExit(f"REGION_MODE must be 'soft', 'all', or 'both' (got {REGION_MODE!r})")
    return w


def main():
    if OUTPUT_MODE not in ("per_patient", "pooled", "both"):
        raise SystemExit(f"OUTPUT_MODE must be 'per_patient', 'pooled', or 'both' (got {OUTPUT_MODE!r})")
    do_pp = OUTPUT_MODE in ("per_patient", "both")
    do_pool = OUTPUT_MODE in ("pooled", "both")
    regions = wanted_regions()

    ct_paths = sorted(glob.glob(os.path.join(ORIG_ROOT, f"{PREFIX}*", CT_NAME)))
    if not ct_paths:
        raise SystemExit(f"No CT files matched: {os.path.join(ORIG_ROOT, PREFIX + '*', CT_NAME)}")
    print(f"Found {len(ct_paths)} patients. OUTPUT_MODE={OUTPUT_MODE}, REGION_MODE={REGION_MODE}")

    os.makedirs(OUT_DIR, exist_ok=True)
    hist_dir = os.path.join(OUT_DIR, "histograms")
    if do_pp:
        os.makedirs(hist_dir, exist_ok=True)

    # pooled accumulators: per region, per stream
    acc = {r: {s: np.zeros(SIZE, dtype=np.int64) for s in ("ct", "orig", "corr")}
           for r in regions} if do_pool else None

    fields = ["patient_id", "region", "n_voxels",
              "ct_p50", "orig_p50", "corr_p50", "shift_orig_ct", "shift_corr_ct"]
    rows = []
    n_used = 0

    for cp in ct_paths:
        pid = os.path.basename(os.path.dirname(cp))
        cbct_o_p = os.path.join(ORIG_ROOT, pid, CBCT_NAME)
        cbct_c_p = os.path.join(CORR_ROOT, pid, CBCT_NAME)
        mask_p = os.path.join(ORIG_ROOT, pid, MASK_NAME) if USE_PROVIDED_MASK else None
        if not os.path.exists(cbct_c_p):
            print(f"  [miss] {pid}: corrected cbct not found")
            continue

        try:
            ct = load_vol(cp); cbct_o = load_vol(cbct_o_p); cbct_c = load_vol(cbct_c_p)
            for nm, v in (("orig", cbct_o), ("corr", cbct_c)):
                if v.shape != ct.shape:
                    raise ValueError(f"shape mismatch ({nm})")
            body = (load_vol(mask_p) > 0) if (mask_p and os.path.exists(mask_p)) else (ct > -500)
            rmask = region_masks(ct, cbct_o, body)

            panels = []
            for r in regions:
                m = rmask[r]
                if m.sum() == 0:
                    continue
                ct_v, o_v, c_v = ct[m], cbct_o[m], cbct_c[m]
                if do_pool:
                    add_to_hist(acc[r]["ct"], ct_v)
                    add_to_hist(acc[r]["orig"], o_v)
                    add_to_hist(acc[r]["corr"], c_v)
                cs, os_, cs2 = stats(ct_v), stats(o_v), stats(c_v)
                rows.append({"patient_id": pid, "region": r, "n_voxels": cs["n"],
                             "ct_p50": cs["p50"], "orig_p50": os_["p50"], "corr_p50": cs2["p50"],
                             "shift_orig_ct": round(os_["mean"] - cs["mean"], 2),
                             "shift_corr_ct": round(cs2["mean"] - cs["mean"], 2)})
                panels.append((r, ct_v, o_v, c_v))

            if do_pp and panels:
                fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 5), squeeze=False)
                for ax, (r, ct_v, o_v, c_v) in zip(axes[0], panels):
                    draw_panel_values(ax, f"{pid} - {r}", ct_v, o_v, c_v)
                fig.tight_layout()
                fig.savefig(os.path.join(hist_dir, f"{pid}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

            n_used += 1
            print(f"  [ok] {pid}")

        except Exception as e:
            print(f"  [error] {pid}: {e}")

    # pooled PNG + ALL rows
    if do_pool and n_used:
        fig, axes = plt.subplots(1, len(regions), figsize=(9 * len(regions), 5), squeeze=False)
        for ax, r in zip(axes[0], regions):
            draw_panel_acc(ax, f"POOLED {PREFIX} - {r}", acc[r]["ct"], acc[r]["orig"], acc[r]["corr"])
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "pooled.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        for r in regions:
            a = {k: hist_stats_acc(acc[r][k]) for k in ("ct", "orig", "corr")}
            if all(a.values()):
                rows.append({"patient_id": "ALL", "region": r, "n_voxels": a["ct"]["n"],
                             "ct_p50": a["ct"]["p50"], "orig_p50": a["orig"]["p50"], "corr_p50": a["corr"]["p50"],
                             "shift_orig_ct": round(a["orig"]["mean"] - a["ct"]["mean"], 2),
                             "shift_corr_ct": round(a["corr"]["mean"] - a["ct"]["mean"], 2)})

    with open(os.path.join(OUT_DIR, "compare_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    print(f"\nDone. {n_used} patients.")
    print(f"CSV -> {os.path.join(OUT_DIR, 'compare_stats.csv')}")
    if do_pool:
        print(f"Pooled PNG -> {os.path.join(OUT_DIR, 'pooled.png')}")
    if do_pp:
        print(f"Per-patient PNGs -> {hist_dir}/<patient>.png")


if __name__ == "__main__":
    main()