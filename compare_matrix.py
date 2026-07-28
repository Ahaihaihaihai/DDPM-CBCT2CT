# -*- coding: utf-8 -*-
"""
compare_matrix.py — Sweep t_start to find the optimum, FAST evaluation (no .npy saved).

Evaluation design (per discussion):
  - PAIRED     : every t_start is tested on the SAME SLICES (fixed seed), and the
                 initial noise is seeded identically per-slice -> differences come
                 purely from t_start.
  - STRATIFIED : N slices are spread evenly across all patients (not piled up).
  - WIN COUNT  : besides the average MAE, count how many slices each t_start "wins"
                 -> a quick way to tell whether the ranking is trustworthy or only
                 a narrow (coincidental) win.

Metrics are computed on the ROI (mask) -> most sensitive for distinguishing t_start.
Does NOT save .npy. Only prints a summary table (that is the deliverable).
"""
import os
import time
import datetime
import random
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

from Diffusion_condition import GaussianDiffusionSampler_cond
from Model_condition import UNet
from datasets import BrainDataset

# ===================== CONFIG =====================
ckpt_trial   = ""                 # <-- checkpoint folder
ckpt_name    = "ckpt_25_.pt"      # <-- checkpoint file name
dataset_name = "../dataset/"

# architecture (match training)
T = 1000; ch = 128; ch_mult = [1, 2, 3, 4]; attn = [2]
num_res_blocks = 2; dropout = 0.3
beta_1 = 1e-4; beta_T = 0.02

T_START_LIST = [900, 920, 950]    # <-- sweep
N_SLICES     = 50                 # number of test slices (spread across patients)
SEED         = 1234               # reproducible: slices & initial noise
HU_MIN, HU_MAX = -1000, 2000
DATA_RANGE   = 4071.0             # PSNR (Liang 2019)
EPS          = 1e-8
# ==================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_hu(a):
    return (a + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN


# ---------- model & sampler ----------
net = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
ckpt_path = f"./Checkpoints/{ckpt_trial}/{ckpt_name}"
net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
net.eval()

sampler = GaussianDiffusionSampler_cond(net, beta_1, beta_T, T).to(device)
sampler.eval()

betas      = torch.linspace(beta_1, beta_T, T).double().to(device)
alphas_bar = torch.cumprod(1.0 - betas, dim=0)
sqrt_ab    = torch.sqrt(alphas_bar).float()
sqrt_1mab  = torch.sqrt(1.0 - alphas_bar).float()


def sample_from_cbct(cbct, t0):
    """CBCT-init (SDEdit) + posterior variance. Identical to Test_condition.py."""
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


# ---------- dataset (test) ----------
ds = BrainDataset(
    root=dataset_name, mode="test", target_size=(256, 256),
    test_max_slices=None, apply_mask_to_image=True,
)


# ---------- slice selection: stratified per patient ----------
def patient_of(idx):
    cbct_path = ds.slices[idx][0]
    return os.path.basename(os.path.dirname(os.path.dirname(cbct_path)))


pat2idx = defaultdict(list)
for i in range(len(ds)):
    pat2idx[patient_of(i)].append(i)

patients = sorted(pat2idx)
per   = N_SLICES // len(patients)
extra = N_SLICES % len(patients)

selected = []
for i, p in enumerate(patients):
    k = per + (1 if i < extra else 0)
    idxs = pat2idx[p]
    if k <= 0:
        continue
    if k >= len(idxs):
        chosen = idxs
    else:
        # evenly spaced -> spread anatomical positions within the patient
        step = len(idxs) / k
        chosen = [idxs[int(j * step + step / 2)] for j in range(k)]
    selected.extend(chosen)

selected = selected[:N_SLICES]
print(f"Testing {len(selected)} slices out of {len(ds)} total, spread across {len(patients)} patients")
print(f"t_start = {T_START_LIST} | seed = {SEED}\n")


# ---------- accumulators ----------
def new_acc():
    return {"sum_abs": 0.0, "sum_sq": 0.0, "n": 0}

acc      = {t0: new_acc() for t0 in T_START_LIST}
acc_base = new_acc()
per_slice_mae = {t0: [] for t0 in T_START_LIST}
win_count = {t0: 0 for t0 in T_START_LIST}


def add(a, err_t):
    a["sum_abs"] += float(err_t.abs().sum().item())
    a["sum_sq"]  += float((err_t.double() ** 2).sum().item())
    a["n"]       += int(err_t.numel())


def mae(a):  return a["sum_abs"] / a["n"] if a["n"] else float("nan")
def rmse(a): return (a["sum_sq"] / a["n"]) ** 0.5 if a["n"] else float("nan")
def psnr(a):
    r = rmse(a)
    return 20.0 * np.log10(DATA_RANGE / r) if r > EPS else float("inf")


# ---------- loop ----------
t0_time = time.time()
for si, idx in enumerate(tqdm(selected, desc="slices")):
    sample   = ds[idx]
    cbct     = sample["CBCT"].unsqueeze(0).to(device)   # (1,1,256,256)
    gt       = sample["pCT"].unsqueeze(0).to(device)
    mask_dev = sample["mask"].unsqueeze(0).to(device)   # (1,1,256,256)
    m        = mask_dev > 0.5

    gt_hu   = to_hu(gt)
    cbct_hu = to_hu(cbct)

    # CBCT baseline (once per slice)
    add(acc_base, (cbct_hu - gt_hu)[m])

    this_mae = {}
    for t0 in T_START_LIST:
        # identical per-slice seed -> same initial noise for every t_start (paired)
        torch.manual_seed(SEED + si)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED + si)

        fake = sample_from_cbct(cbct, t0)
        fake = fake * mask_dev + (-1.0) * (1.0 - mask_dev)   # re-mask
        fake_hu = to_hu(fake)

        err = (fake_hu - gt_hu)[m]
        add(acc[t0], err)
        mae_slice = float(err.abs().mean().item())
        per_slice_mae[t0].append(mae_slice)
        this_mae[t0] = mae_slice

    win_count[min(this_mae, key=this_mae.get)] += 1

dur = datetime.timedelta(seconds=time.time() - t0_time)


# ---------- summary ----------
print(f"\nCBCT baseline (ROI): MAE {mae(acc_base):.2f} | PSNR {psnr(acc_base):.2f}")
print(f"\n===== sCT vs GT (ROI, {len(selected)} slices) =====")
print(f"{'t_start':<10}{'MAE(HU)':>10}{'PSNR(dB)':>10}{'wins':>8}")
print("-" * 38)
for t0 in T_START_LIST:
    avg_slice = sum(per_slice_mae[t0]) / len(per_slice_mae[t0])
    print(f"{t0:<10}{mae(acc[t0]):>10.2f}{psnr(acc[t0]):>10.2f}{win_count[t0]:>8}")

best = min(T_START_LIST, key=lambda t: mae(acc[t]))
print("-" * 38)
print(f"Best (MAE): t_start={best} | wins on {win_count[best]}/{len(selected)} slices")
print(f"Duration: {dur}")
print("\nNote: 'wins' = number of slices where that t_start had the smallest MAE.")
print("  Decisive win (e.g. >70% of slices) = reliable ranking.")
print("  Narrow win (near 50/50) = the average difference is likely coincidental,")
print("  the practical difference is small -> any value is safe to use.")
