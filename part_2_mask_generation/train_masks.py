# train_masks.py
# Train a single-instrument mask with resume + TensorBoard

import os
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

from mask_generation import MASK_GENERATION


# =========================
# Choose target instrument
# =========================
# TARGET_INSTR = "Vx"
# TARGET_INSTR = "Gt"
# TARGET_INSTR = "Bs"
TARGET_INSTR = "Dr"


# =========================
# Hyperparameters & paths
# =========================
MIXTURE_ROOT = f"/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG_By_Instruments/processed_data/stimulus_wav/{TARGET_INSTR}"
SOLO_ROOT    = f"/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG_By_Instruments/processed_data/isolated_wav/{TARGET_INSTR}"

SAVE_DIR   = f"/users/PAS2301/liu215229932/Music_Project/Models/Basen/mask_with_only_audio/masks/{TARGET_INSTR}"
TBOARD_DIR = f"/users/PAS2301/liu215229932/Music_Project/Models/Basen/mask_with_only_audio/tensorboard/{TARGET_INSTR}"

FIXED_LEN     = 837900        # 19s @ 44.1kHz
BATCH_SIZE    = 2
NUM_EPOCHS    = 200
LEARNING_RATE = 1e-3
NUM_WORKERS   = 4
VAL_RATIO     = 0.2
RANDOM_SEED   = 42


# =========================
# Dataset
# =========================
class InstrumentDataset(Dataset):
    def __init__(self, mixture_root, solo_root, fixed_len=None):
        self.mixture_root = mixture_root
        self.solo_root = solo_root
        self.fixed_len = fixed_len
        self.mix_files = sorted([f for f in os.listdir(mixture_root) if f.endswith(".wav")])

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mix_name = self.mix_files[idx]
        mix_path = os.path.join(self.mixture_root, mix_name)
        solo_path = os.path.join(self.solo_root, mix_name.replace("_stimulus", "_soli"))

        mix, _ = sf.read(mix_path)   # (T, 2)
        mix = mix.T                  # -> (2, T)
        solo, _ = sf.read(solo_path) # (T,)

        if self.fixed_len:
            mix = mix[:, :self.fixed_len]
            solo = solo[:self.fixed_len]

        return torch.from_numpy(mix).float(), torch.from_numpy(solo).float()


# =========================
# SI-SDR
# =========================
def si_sdr(pred, target, eps=1e-8):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    s_target = (torch.sum(pred * target, dim=-1, keepdim=True) /
                (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)) * target
    e_noise = pred - s_target
    ratio = (torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)

class SISDRLoss(nn.Module):
    def forward(self, pred, target):
        return -si_sdr(pred, target).mean()


# =========================
# Train (with resume)
# =========================
def train(model, train_loader, val_loader, device,
          num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, save_dir=SAVE_DIR, tb_dir=TBOARD_DIR):

    best_dir   = os.path.join(save_dir, "best_ckpt")
    latest_dir = os.path.join(save_dir, "latest_ckpt")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = SISDRLoss()

    start_epoch = 1
    best_val_sisdr = -float("inf")
    best_train_loss = float("inf")
    latest_ckpt = os.path.join(latest_dir, "model.pt")

    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_sisdr = ckpt.get("val_sisdr", -float("inf"))
        best_train_loss = ckpt.get("best_train_loss", float("inf"))
        print(f"🔄 Resume from epoch {ckpt['epoch']} | last val SI-SDR {best_val_sisdr:.2f} dB")

    # ---- Resume TensorBoard ----
    old_tb_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
    if len(old_tb_files) == 0:
        writer = SummaryWriter(log_dir=tb_dir)
        print("No old TensorBoard logs found, starting fresh.")
    elif len(old_tb_files) == 1:
        old_tb_file = old_tb_files[0]
        print(f"Found old TensorBoard file: {old_tb_file}")
        ea = event_accumulator.EventAccumulator(old_tb_file)
        ea.Reload()
        writer = SummaryWriter(log_dir=tb_dir)
        for tag in ea.Tags()['scalars']:
            for event in ea.Scalars(tag):
                if event.step <= start_epoch - 1:
                    writer.add_scalar(tag, event.value, event.step)
        writer.flush()
        print(f"Re-logged scalars up to epoch {start_epoch-1}")
        os.remove(old_tb_file)
        print(f"Deleted old TensorBoard file. Continuing logging from epoch {start_epoch}")
    else:
        raise RuntimeError(f"Expected 0 or 1 tfevents file in {tb_dir}, found {len(old_tb_files)}")

    # ---- Training loop ----
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running = 0.0
        for mix, solo in train_loader:
            mix, solo = mix.to(device), solo.to(device)
            out = model(mix).squeeze(1)
            loss = criterion(out, solo)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        # ---- Compute training metrics ----
        avg_train_loss = running / len(train_loader)

        # also compute train SI-SDR
        train_sisdr_scores = []
        with torch.no_grad():
            for mix, solo in train_loader:
                mix, solo = mix.to(device), solo.to(device)
                out = model(mix).squeeze(1)
                score = si_sdr(out, solo).mean().item()
                train_sisdr_scores.append(score)
        avg_train_sisdr = float(np.mean(train_sisdr_scores)) if train_sisdr_scores else -float("inf")

        # ---- Validation ----
        model.eval()
        val_losses = []
        val_sisdr_scores = []
        with torch.no_grad():
            for mix, solo in val_loader:
                mix, solo = mix.to(device), solo.to(device)
                out = model(mix).squeeze(1)
                loss_val = criterion(out, solo).item()
                val_losses.append(loss_val)
                score = si_sdr(out, solo).mean().item()
                val_sisdr_scores.append(score)

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        avg_val_sisdr = float(np.mean(val_sisdr_scores)) if val_sisdr_scores else -float("inf")

        # ---- TensorBoard logging ----
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("SI-SDR/train", avg_train_sisdr, epoch)
        writer.add_scalar("SI-SDR/val", avg_val_sisdr, epoch)

        print(f"[{TARGET_INSTR}] Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val SI-SDR: {avg_val_sisdr:.2f} dB")

        # ---- Save latest checkpoint (with epoch number, e.g. latest_ckpt/22.pt) ----
        latest_path = os.path.join(latest_dir, f"{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_sisdr": avg_train_sisdr,
            "val_sisdr": avg_val_sisdr,
        }, latest_path)

        # Remove older checkpoints (keep only the newest one)
        for f in os.listdir(latest_dir):
            fpath = os.path.join(latest_dir, f)
            if f.endswith(".pt") and fpath != latest_path:
                os.remove(fpath)

        # ---- Save best checkpoint (with epoch number only, e.g. 22.pt) ----
        if (avg_val_sisdr > best_val_sisdr) and (avg_train_loss < best_train_loss):
            # Update best metrics
            best_val_sisdr = avg_val_sisdr
            best_train_loss = avg_train_loss

            # Save new best
            best_path = os.path.join(best_dir, f"{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_sisdr": avg_train_sisdr,
                "val_sisdr": avg_val_sisdr,
            }, best_path)
            print(f"✨ Save BEST at {best_path}")

            # Delete old best if different
            for f in os.listdir(best_dir):
                fpath = os.path.join(best_dir, f)
                if fpath != best_path and f.endswith(".pt"):
                    os.remove(fpath)
                    print(f"🗑️ Deleted old best checkpoint: {fpath}")

    print(f"🏁 Training complete. Best Val SI-SDR: {best_val_sisdr:.2f} dB | Best Train Loss: {best_train_loss:.4f}")

    writer.close()


# =========================
# Main
# =========================
if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dataset = InstrumentDataset(MIXTURE_ROOT, SOLO_ROOT, fixed_len=FIXED_LEN)
    n_total = len(dataset)
    n_val = max(1, int(VAL_RATIO * n_total))
    n_train = max(1, n_total - n_val)

    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MASK_GENERATION().to(device)

    train(model, train_loader, val_loader, device,
          num_epochs=NUM_EPOCHS,
          lr=LEARNING_RATE,
          save_dir=SAVE_DIR,
          tb_dir=TBOARD_DIR)
