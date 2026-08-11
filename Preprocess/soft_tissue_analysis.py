"""
CBCT original vs corrected vs CT -- configurable comparison
-----------------------------------------------------------
Three toggles control everything:

  OUTPUT_MODE : "per_patient" | "pooled" | "both"
  REGION_MODE : "soft" | "all" | "both"
  STREAMS     : which curves to compute/plot. Any subset of:
                  "ct"   -> CT
                  "orig" -> CBCT original
                  "corr" -> CBCT corrected
                e.g. ["ct"]  or  ["corr"]  or  ["orig","corr"]  or all three.

The region is defined on the CT (always loaded for masking). No HU value is
shifted/normalized -- read as stored. Pooling uses a 1-HU-bin accumulator.

Outputs (under <OUT_DIR>/):
  compare_stats.csv                 <- one row per (patient/ALL, region, stream)
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
ORIG_ROOT = "/data/3THDD/dataset/CBCT2CT/brain"
CORR_ROOT = "/data/3THDD/dataset/CBCT2CT/corrected_brain"
PREFIX = "2BB"
CT_NAME = "ct.nii.gz"
CBCT_NAME = "cbct.nii.gz"
MASK_NAME = "mask.nii.gz"
USE_PROVIDED_MASK = True

# ---- the three toggles ----
OUTPUT_MODE = "both"           # "per_patient" | "pooled" | "both"
REGION_MODE = "soft"           # "soft" | "all" | "both"
STREAMS = ["orig", "corr", "ct"]   # subset of "ct","orig","corr"

SOFT_LO, SOFT_HI = 0.0, 80.0
CBCT_FLOOR = -900.0

OUT_DIR = f"compare_{PREFIX}"
HIST_BIN_WIDTH = 2.0
HIST_LO, HIST_HI = -1024, 3000
# =======================================================================

SIZE = HIST_HI - HIST_LO + 1
BINS = np.arange(HIST_LO, HIST_HI + 1)

STREAM_LABEL = {"ct": "CT", "orig": "CBCT original", "corr": "CBCT corrected"}
STREAM_COLOR = {"orig": "C0", "corr": "C1", "ct": "C2"}
STREAM_ORDER = ["orig", "corr", "ct"]          # fixed draw order for stable colors


def load_vol(path):
    return nib.load(path).get_fdata().astype(np.float32)


def stats(v):
    p1, p50, p99 = np.percentile(v, [1, 50, 99])
    return {"n": int(v.size), "mean": round(float(v.mean()), 2), "std": round(float(v.std()), 2),
            "p1": round(float(p1), 2), "p50": round(float(p50), 2), "p99": round(float(p99), 2)}


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


def _zoom(stat_list):
    present = [s for s in stat_list if s]
    lo = min(s["p1"] for s in present); hi = max(s["p99"] for s in present)
    pad = 0.05 * (hi - lo) if hi > lo else 10.0
    return lo - pad, hi + pad


def draw_panel_values(ax, label, valdict):
    order = [s for s in STREAM_ORDER if s in valdict]
    lo = min(valdict[s].min() for s in order); hi = max(valdict[s].max() for s in order)
    bins = np.arange(lo, hi + HIST_BIN_WIDTH, HIST_BIN_WIDTH)
    for s in order:
        ax.hist(valdict[s], bins=bins, alpha=0.45, density=True,
                label=STREAM_LABEL[s], color=STREAM_COLOR[s])
    x0, x1 = _zoom([stats(valdict[s]) for s in order])
    ax.set_xlim(x0, x1)
    ax.set_xlabel("HU"); ax.set_ylabel("density"); ax.set_title(label); ax.legend()


def draw_panel_acc(ax, label, accdict):
    order = [s for s in STREAM_ORDER if s in accdict and accdict[s].sum() > 0]
    x0, x1 = _zoom([hist_stats_acc(accdict[s]) for s in order])
    sel = (BINS >= x0) & (BINS <= x1)
    for s in order:
        h = accdict[s]
        ax.plot(BINS[sel], h[sel].astype(np.float64) / h.sum(),
                label=STREAM_LABEL[s], color=STREAM_COLOR[s], linewidth=1.6)
    ax.set_xlabel("HU"); ax.set_ylabel("normalized freq"); ax.set_title(label)
    ax.grid(alpha=0.3); ax.legend()


def region_masks(ct, cbct_o, body):
    valid = cbct_o > CBCT_FLOOR
    region_all = body & valid
    region_soft = region_all & (ct > SOFT_LO) & (ct < SOFT_HI)
    return {"all in-body": region_all, "soft-tissue": region_soft}


def main():
    if OUTPUT_MODE not in ("per_patient", "pooled", "both"):
        raise SystemExit(f"OUTPUT_MODE invalid: {OUTPUT_MODE!r}")
    bad = [s for s in STREAMS if s not in STREAM_LABEL]
    if not STREAMS or bad:
        raise SystemExit(f"STREAMS must be a subset of ['ct','orig','corr'] (bad: {bad})")

    do_pp = OUTPUT_MODE in ("per_patient", "both")
    do_pool = OUTPUT_MODE in ("pooled", "both")
    sel = [s for s in STREAM_ORDER if s in STREAMS]
    load_corr = "corr" in sel

    regions = []
    if REGION_MODE in ("all", "both"):
        regions.append("all in-body")
    if REGION_MODE in ("soft", "both"):
        regions.append("soft-tissue")
    if not regions:
        raise SystemExit(f"REGION_MODE invalid: {REGION_MODE!r}")

    ct_paths = sorted(glob.glob(os.path.join(ORIG_ROOT, f"{PREFIX}*", CT_NAME)))
    if not ct_paths:
        raise SystemExit(f"No CT files matched: {os.path.join(ORIG_ROOT, PREFIX + '*', CT_NAME)}")
    print(f"Found {len(ct_paths)} patients. OUTPUT={OUTPUT_MODE}, REGION={REGION_MODE}, STREAMS={sel}")

    os.makedirs(OUT_DIR, exist_ok=True)
    hist_dir = os.path.join(OUT_DIR, "histograms")
    if do_pp:
        os.makedirs(hist_dir, exist_ok=True)

    acc_streams = sorted(set(sel) | {"ct"})
    acc = ({r: {s: np.zeros(SIZE, dtype=np.int64) for s in acc_streams} for r in regions}
           if do_pool else None)

    fields = ["patient_id", "region", "stream", "n_voxels", "mean", "std",
              "p1", "p50", "p99", "shift_vs_ct"]
    rows = []
    n_used = 0

    for cp in ct_paths:
        pid = os.path.basename(os.path.dirname(cp))
        cbct_o_p = os.path.join(ORIG_ROOT, pid, CBCT_NAME)
        cbct_c_p = os.path.join(CORR_ROOT, pid, CBCT_NAME)
        mask_p = os.path.join(ORIG_ROOT, pid, MASK_NAME) if USE_PROVIDED_MASK else None
        if load_corr and not os.path.exists(cbct_c_p):
            print(f"  [miss] {pid}: corrected cbct not found")
            continue

        try:
            ct = load_vol(cp)
            cbct_o = load_vol(cbct_o_p)
            cbct_c = load_vol(cbct_c_p) if load_corr else None
            for nm, v in (("orig", cbct_o), ("corr", cbct_c)):
                if v is not None and v.shape != ct.shape:
                    raise ValueError(f"shape mismatch ({nm})")
            body = (load_vol(mask_p) > 0) if (mask_p and os.path.exists(mask_p)) else (ct > -500)
            rmask = region_masks(ct, cbct_o, body)

            panels = []
            for r in regions:
                m = rmask[r]
                if m.sum() == 0:
                    continue
                ct_ref = ct[m]
                valdict = {}
                if "ct" in sel:
                    valdict["ct"] = ct_ref
                if "orig" in sel:
                    valdict["orig"] = cbct_o[m]
                if "corr" in sel:
                    valdict["corr"] = cbct_c[m]

                if do_pool:
                    add_to_hist(acc[r]["ct"], ct_ref)
                    if "orig" in sel:
                        add_to_hist(acc[r]["orig"], valdict["orig"])
                    if "corr" in sel:
                        add_to_hist(acc[r]["corr"], valdict["corr"])

                ct_mean = float(ct_ref.mean())
                for s in [x for x in STREAM_ORDER if x in valdict]:
                    st = stats(valdict[s])
                    shift = "" if s == "ct" else round(st["mean"] - ct_mean, 2)
                    rows.append({"patient_id": pid, "region": r, "stream": s,
                                 "n_voxels": st["n"], "mean": st["mean"], "std": st["std"],
                                 "p1": st["p1"], "p50": st["p50"], "p99": st["p99"],
                                 "shift_vs_ct": shift})
                panels.append((r, valdict))

            if do_pp and panels:
                fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 5), squeeze=False)
                for ax, (r, valdict) in zip(axes[0], panels):
                    draw_panel_values(ax, f"{pid} - {r}", valdict)
                fig.tight_layout()
                fig.savefig(os.path.join(hist_dir, f"{pid}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

            n_used += 1
            print(f"  [ok] {pid}")

        except Exception as e:
            print(f"  [error] {pid}: {e}")

    if do_pool and n_used:
        fig, axes = plt.subplots(1, len(regions), figsize=(9 * len(regions), 5), squeeze=False)
        for ax, r in zip(axes[0], regions):
            draw_panel_acc(ax, f"POOLED {PREFIX} - {r}", {s: acc[r][s] for s in sel})
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "pooled.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        for r in regions:
            ct_mean = hist_stats_acc(acc[r]["ct"])["mean"]
            for s in sel:
                a = hist_stats_acc(acc[r][s])
                if a is None:
                    continue
                shift = "" if s == "ct" else round(a["mean"] - ct_mean, 2)
                rows.append({"patient_id": "ALL", "region": r, "stream": s,
                             "n_voxels": a["n"], "mean": a["mean"], "std": a["std"],
                             "p1": a["p1"], "p50": a["p50"], "p99": a["p99"],
                             "shift_vs_ct": shift})

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