#!/usr/bin/env python3
"""
metric_from_npy.py — Metrik sCT vs GT dari .npy hasil Test_condition.py.
PER-SLICE + ringkasan mean±std (per pasien & global). FULL & ROI berdampingan.

Layout input (sesuai Test_condition.py):
    <NPY_ROOT>/<pasien>/sCT/<slice>.npy        -> (3,256,256) HU = [CBCT, Fake, GT]
    <NPY_ROOT>/<pasien>/mask/<slice>_mask.npy  -> (256,256)   0/1
Channel: 0=CBCT, 1=Fake(sCT), 2=GT(CT).

Output:
    - CSV PER PASIEN: <NPY_ROOT>/<pasien>/metrics_per_slice.csv  (1 baris per slice + mean/std)
    - CSV ringkasan : <NPY_ROOT>/metrics_summary.csv             (1 baris per pasien + global)
    - Cetak per-slice ke layar (PRINT_PER_SLICE) + ringkasan.

Pemakaian:
    python metric_from_npy.py
    python metric_from_npy.py <NPY_ROOT>
"""
import sys
import os
import csv
import glob
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

# ============================ CONFIG ============================
NPY_ROOT        = "./test/test/npy"     # <-- root output Test_condition.py
SUMMARY_CSV     = "metrics_summary.csv"  # ditaruh di dalam NPY_ROOT
PRINT_PER_SLICE = True                   # cetak metrik tiap slice ke layar

BG_HU      = -1000.0
DATA_RANGE = 4071.0      # PSNR & SSIM (Liang 2019)
EPS        = 1e-8
CH_CBCT, CH_FAKE, CH_GT = 0, 1, 2
# ===============================================================


def safe_ssim(a, b, data_range):
    h, w = a.shape
    m = min(h, w)
    if m < 3:
        return float("nan")
    win = min(7, m if m % 2 == 1 else m - 1)
    if win < 3:
        win = 3
    return float(ssim_fn(a, b, data_range=data_range, win_size=win))


def slice_metrics(cbct, fake, gt, mask):
    """Metrik untuk SATU slice. Kembalikan dict (full + roi + baseline cbct)."""
    out = {}
    ef = fake - gt

    # ---- FULL (semua pixel, definisi paper eq.23/24) ----
    out["full_mae"]  = float(np.abs(ef).mean())
    r = float(np.sqrt((ef.astype(np.float64) ** 2).mean()))
    out["full_rmse"] = r
    out["full_psnr"] = 20.0 * math.log10(DATA_RANGE / r) if r > EPS else float("inf")
    out["full_ssim"] = safe_ssim(gt, fake, DATA_RANGE)

    # ---- ROI (dalam mask) ----
    if mask.any():
        efm = ef[mask]
        out["roi_mae"]  = float(np.abs(efm).mean())
        rr = float(np.sqrt((efm.astype(np.float64) ** 2).mean()))
        out["roi_rmse"] = rr
        out["roi_psnr"] = 20.0 * math.log10(DATA_RANGE / rr) if rr > EPS else float("inf")
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        out["roi_ssim"] = safe_ssim(gt[y0:y1, x0:x1], fake[y0:y1, x0:x1], DATA_RANGE)
        out["cbct_roi_mae"] = float(np.abs((cbct - gt)[mask]).mean())
        out["roi_n"] = int(mask.sum())
    else:
        for k in ["roi_mae", "roi_rmse", "roi_psnr", "roi_ssim", "cbct_roi_mae"]:
            out[k] = float("nan")
        out["roi_n"] = 0
    return out


def stat(vals):
    a = np.array([v for v in vals
                  if v is not None and not math.isnan(v) and not math.isinf(v)],
                 dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


PER_SLICE_COLS = ["slice",
                  "roi_mae", "roi_rmse", "roi_psnr", "roi_ssim",
                  "full_mae", "full_rmse", "full_psnr", "full_ssim",
                  "cbct_roi_mae", "roi_n"]


def main():
    npy_root = sys.argv[1] if len(sys.argv) > 1 else NPY_ROOT

    patients = sorted([
        d for d in glob.glob(os.path.join(npy_root, "*"))
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "sCT"))
    ])
    if not patients:
        print(f"Tidak ada pasien (folder dgn subfolder sCT/) di: {npy_root}")
        sys.exit(1)

    all_rows = []           # semua slice (untuk global)
    summary_rows = []       # ringkasan per pasien

    for pdir in patients:
        patient = os.path.basename(pdir)
        files = sorted(glob.glob(os.path.join(pdir, "sCT", "*.npy")))
        if not files:
            continue

        rows = []
        if PRINT_PER_SLICE:
            print(f"\n=== {patient} ({len(files)} slice) ===")
            print(f"  {'slice':<16}{'ROI MAE':>9}{'PSNR':>8}{'SSIM':>8}{'CBCT':>9}")

        for f in files:
            slice_name = os.path.splitext(os.path.basename(f))[0]
            st = np.load(f).astype(np.float32)
            cbct, fake, gt = st[CH_CBCT], st[CH_FAKE], st[CH_GT]

            mpath = os.path.join(pdir, "mask", f"{slice_name}_mask.npy")
            mask = (np.load(mpath) > 0.5) if os.path.exists(mpath) else (gt > BG_HU + 1e-3)

            m = slice_metrics(cbct, fake, gt, mask)
            m["slice"] = slice_name
            rows.append(m)
            all_rows.append(m)

            if PRINT_PER_SLICE:
                print(f"  {slice_name:<16}{m['roi_mae']:>9.2f}{m['roi_psnr']:>8.2f}"
                      f"{m['roi_ssim']:>8.3f}{m['cbct_roi_mae']:>9.2f}")

        # ---- CSV per pasien (di folder pasien) ----
        pcsv = os.path.join(pdir, "metrics_per_slice.csv")
        with open(pcsv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(PER_SLICE_COLS)
            for m in rows:
                w.writerow([m["slice"]] + [
                    f"{m[k]:.4f}" if isinstance(m[k], float) else m[k]
                    for k in PER_SLICE_COLS[1:]
                ])
            # baris mean & std
            for label in ("MEAN", "STD"):
                line = [label]
                for k in PER_SLICE_COLS[1:]:
                    if k == "roi_n":
                        line.append("")
                        continue
                    mu, sd = stat([r[k] for r in rows])
                    line.append(f"{mu:.4f}" if label == "MEAN" else f"{sd:.4f}")
                w.writerow(line)

        # ---- ringkasan pasien ----
        pr = {k: stat([r[k] for r in rows]) for k in
              ["roi_mae", "roi_psnr", "roi_ssim", "full_mae", "full_psnr", "cbct_roi_mae"]}
        summary_rows.append((patient, len(rows), pr))
        if PRINT_PER_SLICE:
            print(f"  -> ROI MAE {pr['roi_mae'][0]:.2f}±{pr['roi_mae'][1]:.2f} | "
                  f"PSNR {pr['roi_psnr'][0]:.2f} | SSIM {pr['roi_ssim'][0]:.3f} "
                  f"(CBCT MAE {pr['cbct_roi_mae'][0]:.2f}) | CSV: {pcsv}")

    # ---- CSV ringkasan (per pasien + global) di root ----
    scsv = os.path.join(npy_root, SUMMARY_CSV)
    with open(scsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient", "n_slice", "roi_mae", "roi_psnr", "roi_ssim",
                    "full_mae", "full_psnr", "cbct_roi_mae"])
        for patient, n, pr in summary_rows:
            w.writerow([patient, n,
                        f"{pr['roi_mae'][0]:.4f}", f"{pr['roi_psnr'][0]:.4f}",
                        f"{pr['roi_ssim'][0]:.4f}", f"{pr['full_mae'][0]:.4f}",
                        f"{pr['full_psnr'][0]:.4f}", f"{pr['cbct_roi_mae'][0]:.4f}"])

    # ---- ringkasan GLOBAL (mean±std antar semua slice) ----
    def g(k): return stat([r[k] for r in all_rows])
    print("\n" + "=" * 60)
    print(f"RINGKASAN GLOBAL  ({len(all_rows)} slice, {len(summary_rows)} pasien)")
    print("=" * 60)
    print(f"{'metrik':<16}{'ROI (mask)':>18}{'FULL IMAGE':>18}")
    print("-" * 52)
    print(f"{'MAE (HU)':<16}{g('roi_mae')[0]:>11.2f}±{g('roi_mae')[1]:<5.2f}"
          f"{g('full_mae')[0]:>11.2f}±{g('full_mae')[1]:<5.2f}")
    print(f"{'RMSE (HU)':<16}{g('roi_rmse')[0]:>11.2f}±{g('roi_rmse')[1]:<5.2f}"
          f"{g('full_rmse')[0]:>11.2f}±{g('full_rmse')[1]:<5.2f}")
    print(f"{'PSNR (dB)':<16}{g('roi_psnr')[0]:>11.2f}±{g('roi_psnr')[1]:<5.2f}"
          f"{g('full_psnr')[0]:>11.2f}±{g('full_psnr')[1]:<5.2f}")
    print(f"{'SSIM':<16}{g('roi_ssim')[0]:>11.3f}±{g('roi_ssim')[1]:<5.3f}"
          f"{g('full_ssim')[0]:>11.3f}±{g('full_ssim')[1]:<5.3f}")
    print("-" * 52)
    print(f"CBCT baseline ROI MAE: {g('cbct_roi_mae')[0]:.2f}±{g('cbct_roi_mae')[1]:.2f} HU")
    print(f"Benchmark paper (FULL): MAE ~26 HU | PSNR ~30.5 dB")
    print(f"\nCSV per pasien: <pasien>/metrics_per_slice.csv | ringkasan: {scsv}")


if __name__ == "__main__":
    main()