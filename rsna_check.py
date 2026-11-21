"""Step 5 (optional) - how often are RSNA pneumonia images flagged as TB?

This is a specificity stress test, NOT a pneumonia-vs-TB diagnostic study: RSNA images carry
no TB labels, so every "TB" flag on them is counted as a false alarm for a TB screener.

Prerequisites
  * Accept the RSNA Pneumonia Detection Challenge terms on Kaggle and download it
    (check the competition rules before using or redistributing the data).
  * Use checkpoints trained WITHOUT RSNA data (they are: only Shenzhen/Montgomery are used).

Usage
  python rsna_check.py --rsna_dir /path/to/rsna --ckpt ckpt/resnet50_final.pt ckpt/dinov2_lora_r8_final.pt --n 300
Expected layout in --rsna_dir:
  stage_2_train_images/<patientId>.dcm
  stage_2_detailed_class_info.csv      (columns: patientId, class)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image

from common import DEVICE, EVAL_TF, load_ckpt, preprocess_pil


def read_dicom(path) -> Image.Image:
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a  # invert so air is dark, as in the training images
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return Image.fromarray(a.astype(np.uint8))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


@torch.no_grad()
def tb_probs(model, imgs):
    xs = torch.stack([EVAL_TF(preprocess_pil(i)) for i in imgs]).to(DEVICE)
    return torch.softmax(model(xs).float(), dim=1)[:, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsna_dir", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300, help="images per RSNA class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.rsna_dir)
    info = pd.read_csv(root / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    groups = {c: info[info["class"] == c].sample(args.n, random_state=args.seed)["patientId"].tolist()
              for c in ["Lung Opacity", "Normal"]}  # pneumonia-like opacity vs. normal

    results = []
    for ck in args.ckpt:
        model, tag, thr = load_ckpt(ck)
        for cls, pids in groups.items():
            flags, probs = [], []
            for i in range(0, len(pids), 16):
                imgs = [read_dicom(root / "stage_2_train_images" / f"{p}.dcm") for p in pids[i:i + 16]]
                p = tb_probs(model, imgs)
                probs += p.tolist()
                flags += (p >= thr).tolist()
            k, n = int(sum(flags)), len(flags)
            lo, hi = wilson(k, n)
            results.append(dict(model=tag, rsna_class=cls, n=n, flagged_as_tb=k, flag_rate=k / n,
                                ci_low=lo, ci_high=hi, mean_tb_prob=float(np.mean(probs)), threshold=thr))
    out = pd.DataFrame(results)
    print(out.round(3).to_string(index=False))
    Path("runs").mkdir(exist_ok=True)
    out.to_csv("runs/rsna_false_positive.csv", index=False)


if __name__ == "__main__":
    main()
