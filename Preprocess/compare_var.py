# -*- coding: utf-8 -*-
# FAST comparison: sampling variance -- 'betas' (current code) vs 'posterior' (paper).
# CBCT-init at a fixed t_start, a few slices, then compare MAE & PSNR (ROI).
# Variance is only an inference choice -> no re-training needed.
import os
import numpy as np
import torch
from tqdm import tqdm

from Diffusion_condition import GaussianDiffusionSampler_cond, extract
from Model_condition import UNet
from datasets import BrainDataset

# ===================== CONFIG =====================
CKPT_PATH    = "./Checkpoints/ckpt_5_.pt"   # <-- your trial_9 checkpoint
dataset_name = "../dataset/"
N_SLICES     = 10
T_START      = 400
VAR_MODES    = ["betas", "posterior"]        # 'betas' = current ; 'posterior' = paper
DATA_RANGE   = 4071.0
HU_MIN, HU_MAX = -1000, 2000

T = 1000; ch = 128; ch_mult = [1, 2, 3, 4]; attn = [2]
num_res_blocks = 2; dropout = 0.3
beta_1 = 1e-4; beta_T = 0.02
# ==================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_hu(a):
    return (a + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN


net = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
net.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
net.eval()
sampler = GaussianDiffusionSampler_cond(net, beta_1, beta_T, T).to(device)
sampler.eval()

betas      = torch.linspace(beta_1, beta_T, T).double().to(device)
alphas_bar = torch.cumprod(1.0 - betas, dim=0)
sqrt_ab    = torch.sqrt(alphas_bar).float()
sqrt_1mab  = torch.sqrt(1.0 - alphas_bar).float()

# variance array for each mode
VAR_ARR = {
    "betas":     torch.cat([sampler.posterior_var[1:2], sampler.betas[1:]]),  # current code
    "posterior": sampler.posterior_var,                                       # paper (β̃_t)
}


def sample_cbct(cbct, t0, var_mode):
    var_arr = VAR_ARR[var_mode]
    noise = torch.randn_like(cbct)
    x_ct  = sqrt_ab[t0] * cbct + sqrt_1mab[t0] * noise
    x_t   = torch.cat((x_ct, cbct), dim=1)
    ct    = x_t[:, 0:1]
    cbc   = x_t[:, 1:2]
    with torch.no_grad():
        for ts in reversed(range(t0 + 1)):
            tt  = torch.ones(x_t.shape[0], dtype=torch.long, device=device) * ts
            eps = sampler.model(x_t, tt)
            mean = (extract(sampler.coeff1, tt, ct.shape) * ct
                    - extract(sampler.coeff2, tt, ct.shape) * eps)
            var = extract(var_arr, tt, ct.shape)
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


ds = BrainDataset(root=dataset_name, mode="test", target_size=(256, 256),
                  test_max_slices=None, apply_mask_to_image=True)
idxs = np.linspace(0, len(ds) - 1, N_SLICES).astype(int)
print(f"Testing {len(idxs)} slices (out of {len(ds)} total), t_start={T_START}\n")

acc = {vm: {"mae": [], "psnr": []} for vm in VAR_MODES}
cbct_acc = {"mae": [], "psnr": []}

for idx in tqdm(idxs, desc="slices"):
    sample = ds[int(idx)]
    cbct = sample["CBCT"].unsqueeze(0).to(device)
    gt   = sample["pCT"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)

    gt_hu   = to_hu(gt).squeeze().cpu().numpy()
    cbct_hu = to_hu(cbct).squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()

    mae, psnr = roi_metrics(cbct_hu, gt_hu, mask_np)
    cbct_acc["mae"].append(mae); cbct_acc["psnr"].append(psnr)

    for vm in VAR_MODES:
        fake = sample_cbct(cbct, T_START, vm)
        fake = fake * mask + (-1.0) * (1.0 - mask)
        fake_hu = to_hu(fake).squeeze().cpu().numpy()
        mae, psnr = roi_metrics(fake_hu, gt_hu, mask_np)
        acc[vm]["mae"].append(mae); acc[vm]["psnr"].append(psnr)


def mean(a): return float(np.nanmean(a))

print("\n========== VARIANCE COMPARISON (ROI average) ==========")
print(f"{'method':18s} {'MAE(HU)':>10s} {'PSNR(dB)':>10s}")
print("-" * 40)
print(f"{'CBCT baseline':18s} {mean(cbct_acc['mae']):10.2f} {mean(cbct_acc['psnr']):10.2f}")
for vm in VAR_MODES:
    label = f"var={vm}" + ("  (paper)" if vm == "posterior" else "  (current)")
    print(f"{label:18s} {mean(acc[vm]['mae']):10.2f} {mean(acc[vm]['psnr']):10.2f}")
print("-" * 40)
print("If 'posterior' has much higher MAE / much lower PSNR -> still collapsing, keep using 'betas'.")
print("If 'posterior' is on par / better -> the model is mature enough, you can switch to the paper method.")
