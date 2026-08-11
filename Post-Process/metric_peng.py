#!/usr/bin/env python3
"""
metric_peng.py — Metrics following Junbo Peng et al. (Med Phys 2024), from .npy.

Exact definitions from the paper (full image, eq. 23-25):
  MAE  = (1/N) Σ |sCT - CT|                         (eq. 23)
  PSNR = 10·log10( MAX^2 / MSE ),  MAX = max pixel of sCT & CT   (eq. 24)
  NCC  = (1/N) Σ (sCT-μ_s)(CT-μ_c) / (σ_s·σ_c)       (eq. 25)  [= Pearson correlation]
Computed over the WHOLE image (N = total pixels), as in the paper.
SSIM is NOT in the Peng paper -> included as an optional extra (configurable data_range).

Input layout (Test_condition.py):
  <NPY_ROOT>/<patient>/sCT/<slice>.npy        -> (3,256,256) HU = [CBCT, Fake(sCT), GT]
  <NPY_ROOT>/<patient>/mask/<slice>_mask.npy  -> (256,256) 0/1   (for the optional ROI column)

Output:
  - per-slice print + Table-IV-style summary (CBCT & sCT rows, mean±std)
  - per-patient CSV: <patient>/metrics_peng.csv
  - summary CSV    : <NPY_ROOT>/metrics_peng_summary.csv

Usage: python metric_peng.py [NPY_ROOT]
"""
import sys, os, csv, glob, math
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

# ============================ CONFIG ============================
NPY_ROOT        = "./test/test/npy"
COMPUTE_ROI     = True       # also compute the ROI (in-mask) version for comparison
PRINT_PER_SLICE = True
SSIM_DATA_RANGE = 4071.0     # extra SSIM (not from Peng); team convention
EPS             = 1e-8
CH_CBCT, CH_FAKE, CH_GT = 0, 1, 2
# ===============================================================


def region_metrics(pred, gt, sel=None):
    """MAE, PSNR(Peng), NCC over a region. sel=None -> whole image (Peng's way)."""
    if sel is None:
        p, g = pred.ravel(), gt.ravel()
    else:
        p, g = pred[sel], gt[sel]
    p = p.astype(np.float64); g = g.astype(np.float64)
    if p.size == 0:
        return float("nan"), float("nan"), float("nan")

    mae = float(np.abs(p - g).mean())
    mse = float(((p - g) ** 2).mean())
    vmax = max(p.max(), g.max())
    psnr = 10.0 * math.log10(vmax ** 2 / mse) if (mse > EPS and vmax > 0) else float("inf")

    pm, gm = p - p.mean(), g - g.mean()
    sp, sg = p.std(), g.std()
    ncc = float((pm * gm).mean() / (sp * sg)) if (sp > EPS and sg > EPS) else float("nan")
    return mae, psnr, ncc


def ssim_full(pred, gt, dr):
    h, w = pred.shape
    win = min(7, h if h % 2 == 1 else h - 1)
    if win < 3:
        return float("nan")
    return float(ssim_fn(gt, pred, data_range=dr, win_size=win))


def ssim_roi(pred, gt, mask, dr):
    if not mask.any():
        return float("nan")
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a, b = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]
    m = min(a.shape)
    if m < 3:
        return float("nan")
    win = min(7, m if m % 2 == 1 else m - 1)
    return float(ssim_fn(b, a, data_range=dr, win_size=win))


def stat(vals):
    a = np.array([v for v in vals
                  if v is not None and not math.isnan(v) and not math.isinf(v)],
                 dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def main():
    npy_root = sys.argv[1] if len(sys.argv) > 1 else NPY_ROOT
    patients = sorted([d for d in glob.glob(os.path.join(npy_root, "*"))
                       if os.path.isdir(d) and os.path.isdir(os.path.join(d, "sCT"))])
    if not patients:
        print(f"No patients (folders with an sCT/ subfolder) in: {npy_root}")
        sys.exit(1)

    all_rows = []          # all slices (for global)
    summary = []           # per-patient summary

    cols = ["slice",
            "mae", "psnr", "ncc", "ssim",                      # sCT FULL (Peng)
            "cbct_mae", "cbct_psnr", "cbct_ncc"]               # CBCT FULL (baseline)
    if COMPUTE_ROI:
        cols += ["roi_mae", "roi_psnr", "roi_ncc", "roi_ssim",
                 "roi_cbct_mae", "roi_n"]

    for pdir in patients:
        patient = os.path.basename(pdir)
        files = sorted(glob.glob(os.path.join(pdir, "sCT", "*.npy")))
        if not files:
            continue

        rows = []
        if PRINT_PER_SLICE:
            print(f"\n=== {patient} ({len(files)} slices) === [FULL, Peng's way]")
            print(f"  {'slice':<16}{'MAE':>9}{'PSNR':>8}{'NCC':>8}{'CBCT_MAE':>10}")

        for f in files:
            sname = os.path.splitext(os.path.basename(f))[0]
            st = np.load(f).astype(np.float32)
            cbct, fake, gt = st[CH_CBCT], st[CH_FAKE], st[CH_GT]

            mae, psnr, ncc = region_metrics(fake, gt)                  # sCT FULL
            cmae, cpsnr, cncc = region_metrics(cbct, gt)               # CBCT FULL
            ssim_v = ssim_full(fake, gt, SSIM_DATA_RANGE)

            r = {"slice": sname, "mae": mae, "psnr": psnr, "ncc": ncc, "ssim": ssim_v,
                 "cbct_mae": cmae, "cbct_psnr": cpsnr, "cbct_ncc": cncc}

            if COMPUTE_ROI:
                mpath = os.path.join(pdir, "mask", f"{sname}_mask.npy")
                mask = (np.load(mpath) > 0.5) if os.path.exists(mpath) else (gt > -999.0)
                rmae, rpsnr, rncc = region_metrics(fake, gt, mask)
                rcmae, _, _ = region_metrics(cbct, gt, mask)
                r.update({"roi_mae": rmae, "roi_psnr": rpsnr, "roi_ncc": rncc,
                          "roi_ssim": ssim_roi(fake, gt, mask, SSIM_DATA_RANGE),
                          "roi_cbct_mae": rcmae, "roi_n": int(mask.sum())})

            rows.append(r); all_rows.append(r)
            if PRINT_PER_SLICE:
                print(f"  {sname:<16}{mae:>9.2f}{psnr:>8.2f}{ncc:>8.4f}{cmae:>10.2f}")

        # ---- per-patient CSV (in the patient folder) ----
        pcsv = os.path.join(pdir, "metrics_peng.csv")
        with open(pcsv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(cols)
            for r in rows:
                w.writerow([r["slice"]] + [
                    (f"{r[c]:.4f}" if isinstance(r[c], float) else r[c]) for c in cols[1:]])
            for label in ("MEAN", "STD"):
                line = [label]
                for c in cols[1:]:
                    if c == "roi_n":
                        line.append(""); continue
                    mu, sd = stat([r[c] for r in rows])
                    line.append(f"{mu:.4f}" if label == "MEAN" else f"{sd:.4f}")
                w.writerow(line)

        pr = {c: stat([r[c] for r in rows]) for c in cols[1:] if c != "roi_n"}
        summary.append((patient, len(rows), pr))
        if PRINT_PER_SLICE:
            print(f"  -> sCT FULL: MAE {pr['mae'][0]:.2f} | PSNR {pr['psnr'][0]:.2f} | "
                  f"NCC {pr['ncc'][0]:.4f}  (CBCT MAE {pr['cbct_mae'][0]:.2f}) | CSV: {pcsv}")

    # ---- per-patient summary CSV ----
    scsv = os.path.join(npy_root, "metrics_peng_summary.csv")
    with open(scsv, "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["patient", "n", "mae", "psnr", "ncc", "ssim", "cbct_mae", "cbct_psnr", "cbct_ncc"]
        if COMPUTE_ROI:
            head += ["roi_mae", "roi_psnr", "roi_ncc", "roi_ssim", "roi_cbct_mae"]
        w.writerow(head)
        for patient, n, pr in summary:
            w.writerow([patient, n] + [f"{pr[c][0]:.4f}" for c in head[2:]])

    # ---- Table-IV-style GLOBAL summary (mean±std across slices) ----
    def g(c): return stat([r[c] for r in all_rows])
    def fmt(c): mu, sd = g(c); return f"{mu:.2f}±{sd:.2f}"
    def fmt4(c): mu, sd = g(c); return f"{mu:.4f}±{sd:.4f}"

    print("\n" + "=" * 64)
    print(f"GLOBAL SUMMARY — Peng's way (FULL image)  |  {len(all_rows)} slices")
    print("=" * 64)
    print(f"{'':6}{'MAE (HU)':>16}{'PSNR (dB)':>16}{'NCC':>16}")
    print(f"{'CBCT':6}{fmt('cbct_mae'):>16}{fmt('cbct_psnr'):>16}{fmt4('cbct_ncc'):>16}")
    print(f"{'sCT':6}{fmt('mae'):>16}{fmt('psnr'):>16}{fmt4('ncc'):>16}")
    print(f"\n(extra, not Peng) sCT SSIM FULL: {fmt4('ssim')}")
    if COMPUTE_ROI:
        print("\n--- ROI (in mask) for comparison ---")
        print(f"{'':6}{'MAE (HU)':>16}{'PSNR (dB)':>16}{'NCC':>16}")
        print(f"{'CBCT':6}{fmt('roi_cbct_mae'):>16}{'':>16}{'':>16}")
        print(f"{'sCT':6}{fmt('roi_mae'):>16}{fmt('roi_psnr'):>16}{fmt4('roi_ncc'):>16}")
    print("\nPeng paper benchmark (brain, FULL): MAE 25.99 | PSNR 30.49 | NCC 0.99")
    print("NOTE: the .npy data is clipped to [-1000,2000] HU, so MAX(=max pixel)~2000;")
    print("if Peng did not clip, their PSNR could be slightly higher (mild apples-to-oranges).")
    print(f"\nPer-patient CSV: <patient>/metrics_peng.csv | summary: {scsv}")


if __name__ == "__main__":
    main()
