# -*- coding: utf-8 -*-
"""
test_paper_faithful.py — Test whether the model CAN run from PURE NOISE (the paper's way).

Paper method (Algorithm 2.2, Peng et al.):
    x_T ~ N(0, I)                  # target channel = pure noise (NOT CBCT)
    for t = T-1, ..., 0:           # FULL 1000-step chain
        eps = model(x_t, y=CBCT)   # CBCT concatenated as a CONSTANT condition each step
        x_{t-1} = posterior_mean + sqrt(posterior_var) * z
    return x_0

This DIFFERS from sample_from_cbct (SDEdit) used so far, which initializes from
CBCT-noised to t_start (a team modification, not the paper).

Purpose of the test:
    - If the paper-faithful output is GOOD (~26 HU, clean visuals) -> the setup is
      correct and SDEdit was just a detour. Use the full chain as in the paper.
    - If the output is GARBAGE -> the model did NOT learn to use the CBCT condition
      at high noise. That is a training/conditioning problem, not a matter of finding
      the optimal t_start. SDEdit has been masking this weakness.

For each slice: save a PNG panel + report MAE (ROI). Use only a few slices
(full chain ~2 min/slice as in the paper).
"""
import os
import time
import datetime
import random
from collections import defaultdict
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Diffusion_condition import GaussianDiffusionSampler_cond
from Model_condition import UNet
from datasets import BrainDataset

# ===================== CONFIG =====================
ckpt_trial   = ""                 # <-- checkpoint folder
ckpt_name    = "ckpt_100_.pt"      # <-- checkpoint file name
dataset_name = "../dataset/"

# architecture (match training)
T = 1000; ch = 128; ch_mult = [1, 2, 3, 4]; attn = [2]
num_res_blocks = 2; dropout = 0.3
beta_1 = 1e-4; beta_T = 0.02

N_SLICES        = 6        # only a few (full chain is expensive). Raise if needed.
SEED            = 1234
COMPARE_TSTART  = 400      # SDEdit reference in the panel. None = skip (paper only).
HU_MIN, HU_MAX  = -1000, 2000
WIN_LO, WIN_HI  = -500, 500    # CT display window (same as paper)
ERR_LIM         = 200          # error-map display window [-200,200]
DATA_RANGE      = 4071.0
EPS             = 1e-8

out_dir = "./test/paper_faithful"
# ==================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(out_dir, exist_ok=True)


def to_hu(a):
    return (a + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN


# ---------- model & sampler ----------
net = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
ckpt_path = f"./Checkpoints/{ckpt_trial}/{ckpt_name}"
net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
net.eval()

sampler = GaussianDiffusionSampler_cond(net, beta_1, beta_T, T).to(device)
sampler.eval()

# coefficients for the SDEdit reference
betas      = torch.linspace(beta_1, beta_T, T).double().to(device)
alphas_bar = torch.cumprod(1.0 - betas, dim=0)
sqrt_ab    = torch.sqrt(alphas_bar).float()
sqrt_1mab  = torch.sqrt(1.0 - alphas_bar).float()


def sample_paper(cbct):
    """PAPER-FAITHFUL: start from pure noise, full chain, CBCT as a constant condition."""
    x_ct = torch.randn_like(cbct)          # x_T ~ N(0, I)
    cbc  = cbct
    x_t  = torch.cat((x_ct, cbc), dim=1)
    ct   = x_t[:, 0:1]
    with torch.no_grad():
        for ts in reversed(range(T)):       # 999, 998, ..., 0
            tt = torch.ones(x_t.shape[0], dtype=torch.long, device=device) * ts
            mean, var = sampler.p_mean_variance(x_t=x_t, t=tt)
            z = torch.randn_like(ct) if ts > 0 else 0
            ct = mean + torch.sqrt(var) * z
            x_t = torch.cat((ct, cbc), dim=1)
    return torch.clip(x_t[:, 0:1], -1, 1)


def sample_sdedit(cbct, t0):
    """SDEdit reference: CBCT noised to t0, then denoised. (the old way)"""
    noise = torch.randn_like(cbct)
    x_ct  = sqrt_ab[t0] * cbct + sqrt_1mab[t0] * noise
    cbc   = cbct
    x_t   = torch.cat((x_ct, cbc), dim=1)
    ct    = x_t[:, 0:1]
    with torch.no_grad():
        for ts in reversed(range(t0 + 1)):
            tt = torch.ones(x_t.shape[0], dtype=torch.long, device=device) * ts
            mean, var = sampler.p_mean_variance(x_t=x_t, t=tt)
            z = torch.randn_like(ct) if ts > 0 else 0
            ct = mean + torch.sqrt(var) * z
            x_t = torch.cat((ct, cbc), dim=1)
    return torch.clip(x_t[:, 0:1], -1, 1)


# ---------- dataset & slice selection (stratified) ----------
ds = BrainDataset(
    root=dataset_name, mode="test", target_size=(256, 256),
    test_max_slices=None, apply_mask_to_image=True,
)


def patient_of(idx):
    cbct_path = ds.slices[idx][0]
    return os.path.basename(os.path.dirname(os.path.dirname(cbct_path)))


pat2idx = defaultdict(list)
for i in range(len(ds)):
    pat2idx[patient_of(i)].append(i)

patients = sorted(pat2idx)
rng = random.Random(SEED)
rng.shuffle(patients)

selected = []
for p in patients:
    idxs = pat2idx[p]
    selected.append(idxs[len(idxs) // 2])   # take the middle slice (rich anatomy)
    if len(selected) >= N_SLICES:
        break

print(f"Paper-faithful test on {len(selected)} slices")
print(f"Output panels -> {out_dir}/\n")


def mae_roi(fake_hu, gt_hu, mask_bool):
    return float(np.abs((fake_hu - gt_hu)[mask_bool]).mean())


def disp(ax, img_hu, title, lo=WIN_LO, hi=WIN_HI, cmap="gray"):
    ax.imshow(img_hu, cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ---------- loop ----------
t0_time = time.time()
results = []

for si, idx in enumerate(selected):
    sample = ds[idx]
    patient   = sample["patient"]
    slice_nm  = sample["slice_name"]
    cbct      = sample["CBCT"].unsqueeze(0).to(device)
    gt        = sample["pCT"].unsqueeze(0).to(device)
    mask_dev  = sample["mask"].unsqueeze(0).to(device)
    m_bool    = (mask_dev > 0.5).squeeze().cpu().numpy()

    # paper-faithful
    torch.manual_seed(SEED + si)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + si)
    fake_paper = sample_paper(cbct)
    fake_paper = fake_paper * mask_dev + (-1.0) * (1.0 - mask_dev)

    # SDEdit reference (optional)
    fake_sd = None
    if COMPARE_TSTART is not None:
        torch.manual_seed(SEED + si)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED + si)
        fake_sd = sample_sdedit(cbct, COMPARE_TSTART)
        fake_sd = fake_sd * mask_dev + (-1.0) * (1.0 - mask_dev)

    cbct_hu  = to_hu(cbct).squeeze().cpu().numpy()
    gt_hu    = to_hu(gt).squeeze().cpu().numpy()
    paper_hu = to_hu(fake_paper).squeeze().cpu().numpy()

    mae_p = mae_roi(paper_hu, gt_hu, m_bool)
    mae_c = mae_roi(cbct_hu, gt_hu, m_bool)
    row = {"patient": patient, "slice": slice_nm,
           "mae_cbct": mae_c, "mae_paper": mae_p}

    # ---- panel ----
    ncol = 5 if COMPARE_TSTART is not None else 4
    fig, axes = plt.subplots(1, ncol, figsize=(3 * ncol, 3.2))
    disp(axes[0], cbct_hu, f"CBCT\nMAE {mae_c:.1f}")
    disp(axes[1], paper_hu, f"paper t999\nMAE {mae_p:.1f}")
    c = 2
    if COMPARE_TSTART is not None:
        sd_hu = to_hu(fake_sd).squeeze().cpu().numpy()
        mae_s = mae_roi(sd_hu, gt_hu, m_bool)
        row["mae_sdedit"] = mae_s
        disp(axes[c], sd_hu, f"SDEdit t{COMPARE_TSTART}\nMAE {mae_s:.1f}")
        c += 1
    disp(axes[c], gt_hu, "GT (CT)")
    err = paper_hu - gt_hu
    err[~m_bool] = 0
    axes[c + 1].imshow(err, cmap="bwr", vmin=-ERR_LIM, vmax=ERR_LIM)
    axes[c + 1].set_title("paper - GT", fontsize=9)
    axes[c + 1].axis("off")

    fig.suptitle(f"{patient} / {slice_nm}", fontsize=10)
    fig.tight_layout()
    png = os.path.join(out_dir, f"{patient}_{slice_nm}.png")
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    results.append(row)
    print(f"  [{si+1}/{len(selected)}] {patient}/{slice_nm}  "
          f"paper MAE={mae_p:7.2f} | CBCT={mae_c:7.2f}"
          + (f" | SDEdit={row['mae_sdedit']:7.2f}" if 'mae_sdedit' in row else ""))

dur = datetime.timedelta(seconds=time.time() - t0_time)

# ---------- summary ----------
mean = lambda k: sum(r[k] for r in results) / len(results)
print("\n===== SUMMARY (ROI MAE, HU) =====")
print(f"  CBCT baseline : {mean('mae_cbct'):.2f}")
print(f"  PAPER (t999)  : {mean('mae_paper'):.2f}")
if COMPARE_TSTART is not None:
    print(f"  SDEdit (t{COMPARE_TSTART}) : {mean('mae_sdedit'):.2f}")
print(f"\nDuration: {dur}")
print("\nHow to read:")
print("  - Paper MAE ~26 & clean visuals  -> setup is correct, use the full chain.")
print("  - Paper MAE >> baseline & noisy/garbage visuals -> the model does NOT")
print("    exploit the CBCT condition at high noise. A training/conditioning")
print("    problem, NOT a t_start issue. Check the PNG visuals, not just the numbers.")
