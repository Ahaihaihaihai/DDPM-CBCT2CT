# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
# Convergence version + RESUME (continue / add epochs):
#   - Train & Val loss, curves (train vs val, gap, grad norm, |delta|)
#   - Trainer call is EXACTLY the same: x_0 = cat(CT, CBCT) -> trainer(x_0)
#   - Model-only checkpoint to ./Checkpoints + resume_state.pt to continue training
import os
import re
import csv
import time
import random
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Diffusion_condition import GaussianDiffusionTrainer_cond
from Model_condition import UNet
from datasets import BrainDataset


# ===================== CONFIG =====================
dataset_name = "/data/3THDD/dataset/CBCT2CT/brain_DICOM/"
batch_size = 2
T = 1000
ch = 128
ch_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.3
lr = 1e-4
n_epochs = 200            # final total count. (Model already converges ~epoch 20-30; 200 is excessive.)
beta_1 = 1e-4
beta_T = 0.02
grad_clip = 1
seed = 42

# --- CONTINUE TRAINING (resume) ---
# "" = start from zero. To continue from epoch 100:
#   "./Checkpoints/resume_state.pt" -> perfect resume (if this file exists)
#   "./Checkpoints/ckpt_100_.pt"    -> warm start (weights only; epoch read from name -> start 101)
RESUME_FROM = "./Checkpoints/ckpt_100_.pt"

# --- validation ---
VAL_EVERY      = 1
VAL_MAX_SLICES = 600
VAL_SEED       = 1234

save_weight_dir = "./Checkpoints"
# ==================================================

random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    ws = seed + worker_id
    np.random.seed(ws); random.seed(ws)

g = torch.Generator(); g.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(save_weight_dir, exist_ok=True)

# ---------- data ----------
train_dataloader = DataLoader(
    BrainDataset(root=dataset_name, mode="train", target_size=(256, 256),
                 apply_mask_to_image=True),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
    worker_init_fn=seed_worker, generator=g,
)
val_ds = BrainDataset(root=dataset_name, mode="val", target_size=(256, 256),
                      apply_mask_to_image=True)
if VAL_MAX_SLICES is not None and len(val_ds.slices) > VAL_MAX_SLICES:
    val_ds.slices = random.Random(seed).sample(val_ds.slices, VAL_MAX_SLICES)
    print(f"[val] subset used: {len(val_ds.slices)} slices (fixed seed)")
val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=1, pin_memory=True)

# ---------- model ----------
net_model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
optimizer = torch.optim.AdamW(net_model.parameters(), lr=lr,
                              betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
trainer = GaussianDiffusionTrainer_cond(net_model, beta_1, beta_T, T).to(device)

# ---------- file log ----------
log_path = os.path.join(save_weight_dir, "training.log")
csv_path = os.path.join(save_weight_dir, "loss_log.csv")
png_loss = os.path.join(save_weight_dir, "loss_curve.png")
png_conv = os.path.join(save_weight_dir, "convergence.png")
resume_path = os.path.join(save_weight_dir, "resume_state.pt")
CSV_HEADER = ["epoch", "train_loss", "val_loss", "grad_norm", "duration_sec"]


def load_history_from_csv():
    h = []
    if os.path.exists(csv_path):
        with open(csv_path) as cf:
            for row in csv.DictReader(cf):
                try:
                    h.append({"epoch": int(row["epoch"]),
                              "train": float(row["train_loss"]),
                              "val": float(row["val_loss"]) if row["val_loss"] else None,
                              "gnorm": float(row["grad_norm"])})
                except Exception:
                    pass
    return h


# ---------- resume / fresh ----------
history = []
start_epoch = 1
if RESUME_FROM:
    ckpt = torch.load(RESUME_FROM, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:           # perfect resume
        net_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        history = ckpt.get("history", []) or load_history_from_csv()
        print(f"[resume] perfect resume from epoch {ckpt['epoch']} -> start {start_epoch}")
    else:                                                    # warm start (model-only)
        net_model.load_state_dict(ckpt)
        m = re.search(r"ckpt_(\d+)", os.path.basename(RESUME_FROM))
        last = int(m.group(1)) if m else 0
        start_epoch = last + 1
        history = load_history_from_csv()
        print(f"[warm start] weights from epoch {last} (optimizer reset) -> start {start_epoch}")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as cf:
            csv.writer(cf).writerow(CSV_HEADER)
else:
    with open(csv_path, "w", newline="") as cf:               # fresh -> new CSV
        csv.writer(cf).writerow(CSV_HEADER)

if start_epoch > n_epochs:
    raise SystemExit(f"start_epoch ({start_epoch}) > n_epochs ({n_epochs}). Increase n_epochs.")


@torch.no_grad()
def validate():
    net_model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(VAL_SEED)
    total = 0.0; nb = 0
    for batch in val_dataloader:
        ct   = batch["pCT"].to(device, non_blocking=True)
        cbct = batch["CBCT"].to(device, non_blocking=True)
        total += trainer(torch.cat((ct, cbct), 1)).item(); nb += 1
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    net_model.train()
    return total / max(nb, 1)


def save_curves():
    eps = [h["epoch"] for h in history]; tr = [h["train"] for h in history]
    gn = [h["gnorm"] for h in history]
    v_eps = [h["epoch"] for h in history if h["val"] is not None]
    v_val = [h["val"] for h in history if h["val"] is not None]
    if not eps: return
    plt.figure(figsize=(8, 5))
    plt.plot(eps, tr, marker="o", lw=1.5, label="train")
    if v_val: plt.plot(v_eps, v_val, marker="s", lw=1.5, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE noise)")
    plt.title("Training vs Validation loss"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(png_loss, dpi=120); plt.close()

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(eps, tr, marker="o", label="train")
    if v_val: ax[0, 0].plot(v_eps, v_val, marker="s", label="val")
    ax[0, 0].set_title("Train vs Val loss"); ax[0, 0].set_xlabel("epoch")
    ax[0, 0].grid(alpha=0.3); ax[0, 0].legend()
    if v_val:
        tr_map = {h["epoch"]: h["train"] for h in history}
        gap = [vv - tr_map[e] for e, vv in zip(v_eps, v_val)]
        ax[0, 1].plot(v_eps, gap, marker="o", color="tab:red"); ax[0, 1].axhline(0, color="gray", lw=0.8)
    ax[0, 1].set_title("Gap (val - train)  >0 = starting to overfit"); ax[0, 1].set_xlabel("epoch"); ax[0, 1].grid(alpha=0.3)
    ax[1, 0].plot(eps, gn, marker="o", color="tab:green")
    ax[1, 0].set_title("Grad norm (pre-clip) -- stability"); ax[1, 0].set_xlabel("epoch"); ax[1, 0].grid(alpha=0.3)
    if len(tr) > 1:
        deltas = [abs(tr[i] - tr[i - 1]) for i in range(1, len(tr))]
        ax[1, 1].plot(eps[1:], deltas, marker="o", color="tab:purple"); ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("|delta train loss| (log) -- toward 0 = plateau"); ax[1, 1].set_xlabel("epoch")
    ax[1, 1].grid(alpha=0.3, which="both")
    fig.suptitle("Convergence diagnostics", fontsize=13)
    plt.tight_layout(); plt.savefig(png_conv, dpi=120); plt.close()


# ---------- training ----------
with open(log_path, "a") as log_file:
    log_file.write("\n===== START TRAINING =====\n")
    log_file.write(f"Start epoch: {start_epoch}/{n_epochs} | resume_from='{RESUME_FROM}'\n")
    log_file.write(f"Batch: {batch_size} | LR: {lr} | Seed: {seed} | Device: {device}\n")
    log_file.flush()
    prev_time = time.time()

    for epoch in range(start_epoch, n_epochs + 1):
        net_model.train()
        loss_save = gnorm_save = 0.0; n_batch = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch [{epoch}/{n_epochs}]", unit="batch", ncols=110)
        for batch in pbar:
            optimizer.zero_grad()
            ct   = batch["pCT"].to(device, non_blocking=True)
            cbct = batch["CBCT"].to(device, non_blocking=True)
            loss = trainer(torch.cat((ct, cbct), 1))
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net_model.parameters(), grad_clip)
            optimizer.step()
            loss_save += loss.item(); gnorm_save += float(gnorm); n_batch += 1
            pbar.set_postfix(Loss=f"{loss.item():.4f}")

        avg_loss = loss_save / max(n_batch, 1)
        avg_gnorm = gnorm_save / max(n_batch, 1)
        val_loss = validate() if (epoch % VAL_EVERY == 0 or epoch == n_epochs) else None

        now = time.time(); dur_sec = now - prev_time
        eta = datetime.timedelta(seconds=(n_epochs - epoch) * dur_sec); prev_time = now

        history.append({"epoch": epoch, "train": avg_loss, "val": val_loss, "gnorm": avg_gnorm})
        with open(csv_path, "a", newline="") as cf:
            csv.writer(cf).writerow([epoch, f"{avg_loss:.6f}",
                                     "" if val_loss is None else f"{val_loss:.6f}",
                                     f"{avg_gnorm:.4f}", f"{dur_sec:.2f}"])
        save_curves()

        val_str = "  -" if val_loss is None else f"{val_loss:.4f}"
        msg = (f"-> Epoch {epoch}/{n_epochs} | TrainLoss: {avg_loss:.4f} | ValLoss: {val_str} | "
               f"GradNorm: {avg_gnorm:.3f} | Duration: {datetime.timedelta(seconds=dur_sec)} | ETA: {eta}")
        print(msg); log_file.write(msg + "\n"); log_file.flush()

        if epoch % 5 == 0 or epoch == n_epochs:
            ckpt_path = os.path.join(save_weight_dir, f"ckpt_{epoch}_.pt")
            torch.save(net_model.state_dict(), ckpt_path)
            torch.save({"model": net_model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "history": history}, resume_path)
            print(f"[OK] {ckpt_path} (+ resume_state.pt)")
            log_file.write(f"[OK] {ckpt_path}\n"); log_file.flush()

print(f"\nDone. Curves: {png_loss} & {png_conv} | CSV: {csv_path} | Resume: {resume_path}")