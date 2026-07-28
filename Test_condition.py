# -*- coding: utf-8 -*-
# Test: FULL-CHAIN sampling as in the paper (Algorithm 2.2, Peng et al.).
#   x_T ~ N(0, I) -> full T-step denoising, CBCT concatenated as condition at each step.
#
# Saves PER PATIENT:
#   .npy  -> <npy_root>/<patient>/sCT/<slice>.npy  = (3,256,256) HU [CBCT, sCT, GT]
#            <npy_root>/<patient>/mask/<slice>_mask.npy = (256,256) 0/1
#   PNG   -> <img_root>/<patient>/<slice>.png  = panel [CBCT | sCT | GT], window [-500,500]
#            (same quality/appearance as test_sementara.py)
#
# RESUME=True -> slices whose output already exists will be skipped.
import os
import time
import datetime
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Diffusion_condition import GaussianDiffusionSampler_cond
from Model_condition import UNet
from datasets import BrainDataset

# ===================== CONFIG =====================
out_name     = "test_1"                       # OUTPUT folder
CKPT_PATH    = "./Checkpoints/ckpt_100_.pt"  # checkpoint path (direct, not split)
dataset_name = "../dataset/"

T = 1000; ch = 128; ch_mult = [1, 2, 3, 4]; attn = [2]
num_res_blocks = 2; dropout = 0.3
beta_1 = 1e-4; beta_T = 0.02

RESUME   = True
SEED     = 1234                # per-slice seed -> reproducible results
HU_MIN, HU_MAX = -1000, 2000
WIN_LO, WIN_HI = -500, 500     # PNG display window (same as paper / test_sementara)
SAVE_PNG = False                # save panel [CBCT | sCT | GT]

npy_root = f"./test/{out_name}/npy"
img_root = f"./test/{out_name}/images"
# ==================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(npy_root, exist_ok=True)
os.makedirs(img_root, exist_ok=True)


def to_hu(a):
    return (a + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN


# ---------- model & sampler ----------
net = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
net.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
net.eval()
sampler = GaussianDiffusionSampler_cond(net, beta_1, beta_T, T).to(device)
sampler.eval()


def sample_paper(cbct):
    """FULL-CHAIN (paper): start from pure noise, full T-step denoising,
    CBCT concatenated as a CONSTANT condition at every step (posterior variance)."""
    x_ct = torch.randn_like(cbct)                 # x_T ~ N(0, I)
    cbc  = cbct
    x_t  = torch.cat((x_ct, cbc), dim=1)
    ct   = x_t[:, 0:1]
    with torch.no_grad():
        for ts in reversed(range(T)):              # 999, 998, ..., 0
            tt = torch.ones(x_t.shape[0], dtype=torch.long, device=device) * ts
            mean, var = sampler.p_mean_variance(x_t=x_t, t=tt)
            z = torch.randn_like(ct) if ts > 0 else 0
            ct = mean + torch.sqrt(var) * z
            x_t = torch.cat((ct, cbc), dim=1)
    return torch.clip(x_t[:, 0:1], -1, 1)


def save_panel(png_path, cbct_hu, sct_hu, gt_hu, title):
    """Panel [CBCT | sCT | GT] with window [-500,500] (same as test_sementara)."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, im, t in zip(axes, [cbct_hu, sct_hu, gt_hu],
                         ["CBCT", "sCT (paper t999)", "GT (CT)"]):
        ax.imshow(im, cmap="gray", vmin=WIN_LO, vmax=WIN_HI)
        ax.set_title(t, fontsize=9); ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- dataset (test) ----------
ds = BrainDataset(root=dataset_name, mode="test", target_size=(256, 256),
                  test_max_slices=None, apply_mask_to_image=True)
print(f"Total test slices: {len(ds)}")
print(f"Checkpoint -> {CKPT_PATH}")
print(f"Method: FULL-CHAIN (paper), T={T}, RESUME={RESUME}, SAVE_PNG={SAVE_PNG}\n")


def patient_and_slice(idx):
    cbct_path  = ds.slices[idx][0]
    patient    = os.path.basename(os.path.dirname(os.path.dirname(cbct_path)))
    slice_name = os.path.splitext(os.path.basename(cbct_path))[0]
    return patient, slice_name


# ---------- sampling loop ----------
t0_time = time.time()
saved = skipped = 0

for idx in tqdm(range(len(ds)), desc="sampling"):
    patient, slice_name = patient_and_slice(idx)

    sct_dir  = os.path.join(npy_root, patient, "sCT")
    mask_dir = os.path.join(npy_root, patient, "mask")
    img_dir  = os.path.join(img_root, patient)
    npy_path = os.path.join(sct_dir, f"{slice_name}.npy")
    mask_npy = os.path.join(mask_dir, f"{slice_name}_mask.npy")
    png_path = os.path.join(img_dir, f"{slice_name}.png")

    done_npy = os.path.exists(npy_path) and os.path.exists(mask_npy)
    done_png = (not SAVE_PNG) or os.path.exists(png_path)
    if RESUME and done_npy and done_png:
        skipped += 1
        continue

    os.makedirs(sct_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    if SAVE_PNG:
        os.makedirs(img_dir, exist_ok=True)

    # per-slice seed -> reproducible
    torch.manual_seed(SEED + idx)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + idx)

    sample = ds[idx]
    cbct = sample["CBCT"].unsqueeze(0).to(device)   # (1,1,256,256)
    gt   = sample["pCT"].unsqueeze(0).to(device)
    mask = sample["mask"]                           # (1,256,256)

    fake = sample_paper(cbct)                        # (1,1,256,256) in [-1,1]

    # re-mask sCT -> background -1 (= -1000 HU), consistent with CBCT/GT
    mask_dev = mask.unsqueeze(0).to(device)
    fake = fake * mask_dev + (-1.0) * (1.0 - mask_dev)

    cbct_hu = to_hu(cbct).squeeze().cpu().numpy()
    fake_hu = to_hu(fake).squeeze().cpu().numpy()    # = sCT
    gt_hu   = to_hu(gt).squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()

    stacked = np.stack([cbct_hu, fake_hu, gt_hu], axis=0).astype(np.float32)  # [CBCT, sCT, GT]
    np.save(npy_path, stacked)
    np.save(mask_npy, mask_np.astype(np.float32))

    if SAVE_PNG:
        save_panel(png_path, cbct_hu, fake_hu, gt_hu, f"{patient} / {slice_name}")

    saved += 1

dur = datetime.timedelta(seconds=time.time() - t0_time)
print(f"\nDone. Saved: {saved} | Skipped (resume): {skipped} | Duration: {dur}")
print(f".npy -> {npy_root}/<patient>/sCT/   |   PNG -> {img_root}/<patient>/")
