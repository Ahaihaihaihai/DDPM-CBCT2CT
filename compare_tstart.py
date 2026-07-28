# -*- coding: utf-8 -*-
# FAST comparison: run CBCT-init sampling at several t_start values on a few slices,
# then compare MAE & PSNR (ROI) vs the CBCT baseline.
# t_start=999 ~= sampling from pure noise (similar to the paper's Algorithm 2.2).
import os
import numpy as np
import torch
from tqdm import tqdm

from Diffusion_condition import GaussianDiffusionSampler_cond
from Model_condition import UNet
from datasets import BrainDataset

# ===================== CONFIG =====================
CKPT_PATH    = "./Checkpoints/ckpt_5_.pt"   # <-- your checkpoint path
dataset_name = "../dataset/"
N_SLICES     = 10                       # number of test slices (evenly sampled from the test set)
T_START_LIST = [250, 400, 700, 999]          # t_start values to compare
DATA_RANGE   = 4071.0
HU_MIN, HU_MAX = -1000, 2000

# architecture (match training)
T = 1000; ch = 128; ch_mult = [1, 2, 3, 4]; attn = [2]
num_res_blocks = 2; dropout = 0.3
beta_1 = 1e-4; beta_T = 0.02
# ==================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_hu(a):
    return (a + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN


# ---------- model & sampler ----------
net = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
net.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
net.eval()
sampler = GaussianDiffusionSampler_cond(net, beta_1, beta_T, T).to(device)
sampler.eval()

betas      = torch.linspace(beta_1, beta_T, T).double().to(device)
alphas_bar = torch.cumprod(1.0 - betas, dim=0)
sqrt_ab    = torch.sqrt(alphas_bar).float()
sqrt_1mab  = torch.sqrt(1.0 - alphas_bar).float()
print(f"sqrt_ab[400]={sqrt_ab[400]:.4f}  sqrt_ab[999]={sqrt_ab[999]:.4f}  "
      f"(smaller = closer to pure noise)\n")


def sample_from_cbct(cbct, t0):
    noise = torch.randn_like(cbct)
    x_ct  = sqrt_ab[t0] * cbct + sqrt_1mab[t0] * noise
    x_t   = torch.cat((x_ct, cbct), dim=1)
    ct    = x_t[:, 0:1]
    cbc   = x_t[:, 1:2]
    with torch.no_grad():
        for ts in reversed(range(t0 + 1)):
            tt = torch.ones(x_t.shape[0], dtype=torch.long, device=device) * ts
            mean, var = sampler.p_mean_variance(x_t=x_t, t=tt)
            z = torch.randn_like(ct) if ts > 0 else 0
            ct = mean + torch.sqrt(var) * z
            x_t = torch.cat((ct, cbc), dim=1)
    return torch.clip(x_t[:, 0:1], -1, 1)


def roi_metrics(pred_hu, gt_hu, mask_np):
    m = mask_np > 0
    if m.sum() == 0:
        return np.nan, np.nan
    diff = pred_hu[m] - gt_hu[m]
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    psnr = 20.0 * np.log10(DATA_RANGE / rmse) if rmse > 0 else np.inf
    return mae, psnr


# ---------- data ----------
ds = BrainDataset(root=dataset_name, mode="test", target_size=(256, 256),
                  test_max_slices=None, apply_mask_to_image=True)
idxs = np.linspace(0, len(ds) - 1, N_SLICES).astype(int)
print(f"Testing {len(idxs)} slices (out of {len(ds)} total test).\n")

acc = {t0: {"mae": [], "psnr": []} for t0 in T_START_LIST}
cbct_acc = {"mae": [], "psnr": []}

for idx in tqdm(idxs, desc="slices"):
    sample = ds[int(idx)]
    cbct = sample["CBCT"].unsqueeze(0).to(device)
    gt   = sample["pCT"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)   # (1,1,256,256)

    gt_hu   = to_hu(gt).squeeze().cpu().numpy()
    cbct_hu = to_hu(cbct).squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()

    mae, psnr = roi_metrics(cbct_hu, gt_hu, mask_np)
    cbct_acc["mae"].append(mae); cbct_acc["psnr"].append(psnr)

    for t0 in T_START_LIST:
        fake = sample_from_cbct(cbct, t0)
        fake = fake * mask + (-1.0) * (1.0 - mask)   # re-mask as in your pipeline
        fake_hu = to_hu(fake).squeeze().cpu().numpy()
        mae, psnr = roi_metrics(fake_hu, gt_hu, mask_np)
        acc[t0]["mae"].append(mae); acc[t0]["psnr"].append(psnr)


def mean(a): return float(np.nanmean(a))

print("\n========== COMPARISON (ROI average) ==========")
print(f"{'method':14s} {'MAE(HU)':>10s} {'PSNR(dB)':>10s}")
print("-" * 36)
print(f"{'CBCT baseline':14s} {mean(cbct_acc['mae']):10.2f} {mean(cbct_acc['psnr']):10.2f}")
for t0 in T_START_LIST:
    label = f"t_start={t0}"
    print(f"{label:14s} {mean(acc[t0]['mae']):10.2f} {mean(acc[t0]['psnr']):10.2f}")
print("-" * 36)
print("Lower MAE is better | Higher PSNR is better")
