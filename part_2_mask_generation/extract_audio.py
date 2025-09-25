# extract_all.py
# Iterate over global stimulus_wav folder, match target list,
# run the correct instrument mask, save extracted audio + SI-SDR

import os
import torch
import soundfile as sf
import numpy as np
from mask_generation import MASK_GENERATION
from train_masks import si_sdr   # reuse SI-SDR function


# =========================
# Config
# =========================
STIMULUS_ROOT = "/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/stimulus_wav"
SOLO_ROOT     = "/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/isolated_wav"
MASKS_ROOT    = "/users/PAS2301/liu215229932/Music_Project/Models/Basen/mask_with_only_audio/masks"
OUTPUT_ROOT   = "/users/PAS2301/liu215229932/Music_Project/Models/Basen/mask_with_only_audio/extracted_audio"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

FILES_TO_EXTRACT = [
    "0001_pop_falldead_duo_BsDr_theme1_stereo_Bs",
    "0003_pop_falldead_duo_BsDr_theme1_stereo_Bs",
    "0003_pop_falldead_duo_GtVx_theme1_stereo_Gt",
    "0003_pop_falldead_trio_GtDrVx_theme1_stereo_Dr",
    "0003_pop_falldead_trio_GtDrVx_theme1_stereo_Gt",
    "0003_pop_falldead_trio_GtDrVx_theme1_stereo_Vx",
    "0003_pop_mixtape_duo_BsDr_theme1_stereo_Bs",
    "0003_pop_mixtape_duo_BsDr_theme1_stereo_Dr",
    "0003_pop_mixtape_duo_GtVx_theme1_stereo_Vx",
    "0003_pop_mixtape_duo_GtVx_theme2_stereo_Gt",
    "0003_pop_mixtape_trio_BsDrVx_theme1_stereo_Vx",
    "0003_pop_mixtape_trio_GtDrVx_theme1_stereo_Gt",
    "0003_pop_mixtape_trio_GtDrVx_theme2_stereo_Dr",
    "0007_pop_mixtape_duo_BsDr_theme1_stereo_Bs",
    "0007_pop_mixtape_duo_BsDr_theme1_stereo_Dr",
    "0007_pop_mixtape_duo_GtVx_theme1_stereo_Vx",
    "0007_pop_mixtape_duo_GtVx_theme2_stereo_Gt",
    "0007_pop_mixtape_trio_BsDrVx_theme2_stereo_Bs",
    "0007_pop_mixtape_trio_BsDrVx_theme2_stereo_Dr",
    "0007_pop_mixtape_trio_GtDrVx_theme2_stereo_Vx",
]

VALID_INSTRS = ["Gt", "Vx", "Bs", "Dr"]


# =========================
# Load all models once
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}

for instr in VALID_INSTRS:
    best_dir = os.path.join(MASKS_ROOT, instr, "best_ckpt")
    best_files = [f for f in os.listdir(best_dir) if f.endswith(".pt")]
    assert len(best_files) == 1, f"Expected 1 best ckpt for {instr}, found {len(best_files)}"
    ckpt_path = os.path.join(best_dir, best_files[0])

    model = MASK_GENERATION().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    models[instr] = model
    print(f"✅ Loaded {instr} model from {ckpt_path}")


# =========================
# Main loop
# =========================
results = []

for fname in os.listdir(STIMULUS_ROOT):
    if not fname.endswith("_stimulus.wav"):
        continue
    base = fname.replace("_stimulus.wav", "")
    if base not in FILES_TO_EXTRACT:
        continue

    instr = base.split("_")[-1]
    if instr not in VALID_INSTRS:
        print(f"⚠️ Skipping {fname}, instrument {instr} not in {VALID_INSTRS}")
        continue

    # build paths
    mix_path = os.path.join(STIMULUS_ROOT, fname)
    solo_path = os.path.join(SOLO_ROOT, base + "_soli.wav")
    out_dir = os.path.join(OUTPUT_ROOT, instr)
    os.makedirs(out_dir, exist_ok=True)

    # load audio
    mix, sr = sf.read(mix_path)   # (T, 2)
    solo, _ = sf.read(solo_path) # (T,)
    mix_tensor = torch.from_numpy(mix.T).unsqueeze(0).float().to(device)

    # run model
    model = models[instr]
    with torch.no_grad():
        est = model(mix_tensor).squeeze().cpu().numpy()

    # save audio
    out_path = os.path.join(out_dir, base + "_extracted.wav")
    sf.write(out_path, est, sr)
    print(f"💾 Saved {out_path}")

    # compute SI-SDR
    est_tensor = torch.from_numpy(est).unsqueeze(0).to(device)
    solo_tensor = torch.from_numpy(solo).unsqueeze(0).to(device)
    score = si_sdr(est_tensor, solo_tensor).item()
    results.append((base, instr, score))
    print(f"📊 SI-SDR {base} ({instr}): {score:.2f} dB")


# =========================
# Save all scores
# =========================
score_path = os.path.join(OUTPUT_ROOT, "SI-SDR_Score.txt")
with open(score_path, "w") as f:
    for base, instr, score in results:
        f.write(f"{base} ({instr}): {score:.2f} dB\n")
print(f"✅ Scores saved to {score_path}")
