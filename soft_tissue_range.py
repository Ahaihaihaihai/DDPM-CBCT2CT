# -*- coding: utf-8 -*-
# ============================================================================
# soft_tissue_range.py
#
# Research the SOFT-TISSUE HU distribution range per CENTER (2BB, 2BC, ...).
# Reads the RAW DICOM directly (applies RescaleSlope/Intercept AS RECORDED),
# so per-center CBCT calibration differences stay visible -- this is exactly
# what we want to study (e.g. 2BA intercept=0, 2BB=-1000, 2BC=-1024).
#
# For each image it:
#   1. builds a BODY mask -- uses the patient's mask/ folder if present,
#      otherwise a threshold fallback (2BB / 2BC have no mask/ folder).
#   2. keeps the in-body voxels and (separately) a soft-tissue window.
#   3. accumulates a 1-HU-bin histogram -> exact percentiles, low memory.
#
# Output:
#   - per-patient + per-center printed tables
#   - CSV : ./soft_tissue_range/soft_tissue_stats.csv
#   - PNG : ./soft_tissue_range/<center>_hist.png  (one curve per modality)
#
# Does NOT need a checkpoint or torch. Only numpy + pydicom (+ matplotlib for PNG).
# scipy is optional: if installed, the threshold body mask is cleaned up
# (largest connected component + hole filling); otherwise a plain threshold.
# ============================================================================
import os
import csv
import glob
import numpy as np
import pydicom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ===================== CONFIG =====================
DATASET_ROOT = "../dataset/"
CENTERS      = ["2BB", "2BC"]          # patient-ID prefixes to study
MODALITIES   = ["cbct", "cbct_cal", "ct"]   # subfolders analyzed if present

SOFT_LO, SOFT_HI = -300.0, 300.0       # soft-tissue window (HU) for the windowed stats
BODY_THRESHOLD   = -500.0              # HU above this = "inside body" (threshold fallback)
USE_PROVIDED_MASK = True               # use the mask/ folder when available
SLICE_STEP        = 1                  # >1 to subsample slices for speed

# force the CBCT intercept instead of the recorded tag (None = use tag = recommended
# for this study, so the raw per-center calibration is preserved). Only affects
# folders whose name starts with "cbct".
FORCE_CBCT_INTERCEPT = None            # e.g. -1024.0 to normalize; None = as recorded

OUT_DIR   = "./soft_tissue_range"
HIST_LO, HIST_HI = -1024, 3000         # histogram support (HU), 1 HU bins
PLOT_LO, PLOT_HI = -600, 600           # x-range for the PNG plots
# ==================================================

SIZE    = HIST_HI - HIST_LO + 1
CENTERS_ARR = np.arange(HIST_LO, HIST_HI + 1)   # bin center = HU value


# --------------------------- DICOM I/O ---------------------------
def load_hu(path, force_intercept=None):
    """Read one DICOM -> float32 HU array (arr * slope + intercept)."""
    dcm = pydicom.dcmread(path)
    arr = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    if force_intercept is not None:
        intercept = float(force_intercept)
    else:
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def load_mask(path):
    dcm = pydicom.dcmread(path)
    return dcm.pixel_array.astype(np.float32) > 0


# --------------------------- masking ---------------------------
def body_mask_from_threshold(hu):
    """Fallback body mask when no mask/ folder exists."""
    m = hu > BODY_THRESHOLD
    if HAVE_SCIPY:
        m = ndi.binary_fill_holes(m)
        lbl, num = ndi.label(m)
        if num > 1:                      # keep the largest blob (the patient, not the table)
            sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, num + 1))
            m = lbl == (int(np.argmax(sizes)) + 1)
    return m


# --------------------------- histogram stats ---------------------------
def add_to_hist(hist, vals):
    if vals.size == 0:
        return
    idx = np.clip(np.rint(vals).astype(np.int64) - HIST_LO, 0, SIZE - 1)
    hist += np.bincount(idx, minlength=SIZE)


def hist_stats(hist, lo=HIST_LO, hi=HIST_HI):
    """Exact stats from a 1-HU-bin histogram, restricted to [lo, hi]."""
    sel = (CENTERS_ARR >= lo) & (CENTERS_ARR <= hi)
    c = CENTERS_ARR[sel].astype(np.float64)
    h = hist[sel].astype(np.float64)
    n = h.sum()
    if n == 0:
        return None
    mean = float((c * h).sum() / n)
    std  = float(np.sqrt(((c - mean) ** 2 * h).sum() / n))
    cum  = np.cumsum(h)

    def pct(p):
        k = p / 100.0 * n
        i = int(np.searchsorted(cum, k))
        return float(c[min(i, len(c) - 1)])

    nz = np.nonzero(h)[0]
    return {
        "n": int(n), "mean": mean, "std": std,
        "min": float(c[nz[0]]), "max": float(c[nz[-1]]),
        "p1": pct(1), "p5": pct(5), "p25": pct(25), "p50": pct(50),
        "p75": pct(75), "p95": pct(95), "p99": pct(99),
    }


# --------------------------- per-patient processing ---------------------------
def process_modality(patient_dir, modality):
    """Return an in-body 1-HU histogram for one patient/modality (or None)."""
    mod_dir = os.path.join(patient_dir, modality)
    if not os.path.isdir(mod_dir):
        return None
    files = sorted(glob.glob(os.path.join(mod_dir, "*.dcm")))[::SLICE_STEP]
    if not files:
        return None

    # optional provided masks, matched by filename stem
    mask_by_stem = {}
    mask_dir = os.path.join(patient_dir, "mask")
    if USE_PROVIDED_MASK and os.path.isdir(mask_dir):
        for mf in glob.glob(os.path.join(mask_dir, "*.dcm")):
            mask_by_stem[os.path.splitext(os.path.basename(mf))[0]] = mf

    force = FORCE_CBCT_INTERCEPT if modality.startswith("cbct") else None
    hist = np.zeros(SIZE, dtype=np.int64)
    used = 0
    for f in files:
        hu = load_hu(f, force_intercept=force)
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in mask_by_stem:
            m = load_mask(mask_by_stem[stem])
            if m.shape != hu.shape:      # geometry mismatch -> fall back
                m = body_mask_from_threshold(hu)
        else:
            m = body_mask_from_threshold(hu)
        add_to_hist(hist, hu[m])
        used += 1
    return hist if used else None


def patient_center(name):
    return name[:3]


# --------------------------- main ---------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_dirs = sorted(d for d in glob.glob(os.path.join(DATASET_ROOT, "*")) if os.path.isdir(d))

    if not HAVE_SCIPY:
        print("[note] scipy not installed -> body mask is a plain HU threshold "
              "(couch/table may leak in; the soft-tissue window mitigates this).\n")

    center_hist = {c: {m: np.zeros(SIZE, dtype=np.int64) for m in MODALITIES} for c in CENTERS}
    rows = []   # CSV rows

    def fmt(s):
        return (f"n={s['n']:>9d}  mean={s['mean']:7.1f}  std={s['std']:6.1f}  "
                f"[p1 {s['p1']:6.0f} | p25 {s['p25']:6.0f} | p50 {s['p50']:6.0f} | "
                f"p75 {s['p75']:6.0f} | p99 {s['p99']:6.0f}]")

    for center in CENTERS:
        pdirs = [d for d in all_dirs if patient_center(os.path.basename(d)) == center]
        print("=" * 100)
        print(f"CENTER {center}  ({len(pdirs)} patients: {[os.path.basename(d) for d in pdirs]})")
        print("=" * 100)
        if not pdirs:
            print("  (no patients found)\n")
            continue

        for pdir in pdirs:
            pid = os.path.basename(pdir)
            for mod in MODALITIES:
                hist = process_modality(pdir, mod)
                if hist is None:
                    continue
                center_hist[center][mod] += hist
                body = hist_stats(hist)
                soft = hist_stats(hist, SOFT_LO, SOFT_HI)
                if body is None:
                    continue
                soft_frac = (soft["n"] / body["n"]) if soft else 0.0
                print(f"  {pid:<8} {mod:<9} in-body : {fmt(body)}")
                if soft:
                    print(f"  {'':<8} {'':<9} soft{int(SOFT_LO)}..{int(SOFT_HI)}: {fmt(soft)}  "
                          f"(soft frac {soft_frac:5.1%})")
                rows.append(_row(center, pid, mod, body, soft, soft_frac))
            if any(os.path.isdir(os.path.join(pdir, m)) for m in MODALITIES):
                print()

        # ----- per-center summary -----
        print(f"----- {center} CENTER SUMMARY (all patients pooled) -----")
        for mod in MODALITIES:
            hist = center_hist[center][mod]
            body = hist_stats(hist)
            if body is None:
                continue
            soft = hist_stats(hist, SOFT_LO, SOFT_HI)
            soft_frac = (soft["n"] / body["n"]) if soft else 0.0
            print(f"  {mod:<9} in-body : {fmt(body)}")
            if soft:
                print(f"  {'':<9} soft{int(SOFT_LO)}..{int(SOFT_HI)}: {fmt(soft)}  "
                      f"(soft frac {soft_frac:5.1%})")
            rows.append(_row(center, "ALL", mod, body, soft, soft_frac))
        print()

    _save_csv(rows)
    _save_plots(center_hist)

    print("How to read (soft-tissue distribution range):")
    print(f"  - The soft-tissue peak lives inside [{int(SOFT_LO)}, {int(SOFT_HI)}] HU; the")
    print("    p25..p75 band is the practical soft-tissue range for that center/modality.")
    print("  - Compare the SAME modality across centers: if the soft-tissue mean/p50 is")
    print("    shifted (e.g. 2BB vs 2BC on 'cbct'), that is a per-center HU calibration")
    print("    offset -> justifies forcing a common CBCT intercept (see datasets.py).")
    print("  - Compare 'cbct' vs 'cbct_cal' vs 'ct' within a center to see whether the")
    print("    calibrated CBCT already matches CT soft tissue.")
    print(f"\nCSV -> {os.path.join(OUT_DIR, 'soft_tissue_stats.csv')} | plots -> {OUT_DIR}/<center>_hist.png")


def _row(center, pid, mod, body, soft, soft_frac):
    r = {"center": center, "patient": pid, "modality": mod}
    for k in ("n", "mean", "std", "min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"):
        r[k] = body[k]
    r["soft_frac"] = round(soft_frac, 4)
    r["soft_mean"] = soft["mean"] if soft else ""
    r["soft_p50"]  = soft["p50"] if soft else ""
    return r


def _save_csv(rows):
    if not rows:
        return
    path = os.path.join(OUT_DIR, "soft_tissue_stats.csv")
    cols = ["center", "patient", "modality", "n", "mean", "std", "min",
            "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max",
            "soft_frac", "soft_mean", "soft_p50"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.2f}" if isinstance(r[c], float) else r[c]) for c in cols})


def _save_plots(center_hist):
    sel = (CENTERS_ARR >= PLOT_LO) & (CENTERS_ARR <= PLOT_HI)
    x = CENTERS_ARR[sel]
    for center, mods in center_hist.items():
        if not any(mods[m].sum() for m in mods):
            continue
        plt.figure(figsize=(8, 4.5))
        for mod in mods:
            h = mods[mod][sel].astype(np.float64)
            if h.sum() == 0:
                continue
            plt.plot(x, h / h.sum(), label=mod, linewidth=1.4)   # normalized density
        plt.axvspan(SOFT_LO, SOFT_HI, color="grey", alpha=0.12, label=f"soft [{int(SOFT_LO)},{int(SOFT_HI)}]")
        plt.title(f"{center} — in-body HU distribution (pooled patients)")
        plt.xlabel("HU"); plt.ylabel("normalized frequency")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"{center}_hist.png")
        plt.savefig(out, dpi=120); plt.close()


if __name__ == "__main__":
    main()
