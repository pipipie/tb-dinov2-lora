"""Step 4 - Grad-CAM heatmaps (works for ResNet, DenseNet and DINOv2 +/- LoRA).

Usage
  # paper figure: Shenzhen-trained model looking at Montgomery images (never seen in training)
  python explain.py --ckpt ckpt/dinov2_lora_r8_cross_shenzhen.pt --source montgomery --n 3 --out figs/cam_lora.png
  python explain.py --ckpt ckpt/resnet50_cross_shenzhen.pt       --source montgomery --n 3 --out figs/cam_resnet.png
  # your own images
  python explain.py --ckpt ckpt/resnet50_final.pt --images a.png b.png --out figs/cam_custom.png

Grad-CAM shows where the model's evidence is; it is NOT proof the model uses lung pathology.
Look for failure modes (text markers, borders, devices) and discuss them in the paper.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import DEVICE, EVAL_TF, IMG_SIZE, MANIFEST, load_ckpt, load_gray8, parse_tag, preprocess_pil


def _target_layer(model, tag):
    name, _ = parse_tag(tag)
    if name == "resnet50":
        return model.layer4[-1]
    if name == "densenet121":
        return model.features.norm5
    backbone = model.backbone
    inner = backbone.base_model.model if hasattr(backbone, "peft_config") else backbone  # unwrap PEFT
    return inner.encoder.layer[-1].norm1


def _vit_reshape(t):
    # (B, 1 + 16*16, C) -> (B, C, 16, 16): drop CLS token, restore the patch grid
    g = IMG_SIZE // 14
    return t[:, 1:, :].reshape(t.size(0), g, g, t.size(2)).permute(0, 3, 1, 2)


def gradcam(model, tag, pil_img, thr, target_class=1, device=DEVICE):
    """Return (overlay uint8 HxWx3, tb_probability, predicted_positive)."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    img = preprocess_pil(pil_img)
    x = EVAL_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x).float(), dim=1)[0, 1].item()
    x.requires_grad_(True)  # needed when the backbone is frozen (linear probe)
    reshape = _vit_reshape if tag.startswith("dinov2") else None
    with GradCAM(model=model, target_layers=[_target_layer(model, tag)], reshape_transform=reshape) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])[0]
    overlay = show_cam_on_image(np.asarray(img, dtype=np.float32) / 255.0, heat, use_rgb=True)
    return overlay, prob, prob >= thr


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--source", choices=["shenzhen", "montgomery"], default="montgomery")
    ap.add_argument("--n", type=int, default=3, help="examples per class when using --source")
    ap.add_argument("--out", default="figs/gradcam.png")
    args = ap.parse_args()

    model, tag, thr = load_ckpt(args.ckpt)
    if args.images:
        items = [(p, None) for p in args.images]
    else:
        m = pd.read_csv(MANIFEST)
        m = m[m.source == args.source]
        items = [(r.path, r.label) for lbl in (1, 0) for r in m[m.label == lbl].sample(args.n, random_state=0).itertuples()]

    fig, axes = plt.subplots(2, len(items), figsize=(2.6 * len(items), 5.4), squeeze=False)
    for j, (path, label) in enumerate(items):
        pil = load_gray8(path)
        overlay, prob, pos = gradcam(model, tag, pil, thr)
        axes[0, j].imshow(preprocess_pil(pil))
        axes[1, j].imshow(overlay)
        truth = "" if label is None else f"true: {'TB' if label else 'normal'}\n"
        axes[1, j].set_title(f"{truth}P(TB)={prob:.2f} -> {'TB' if pos else 'normal'}", fontsize=8)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(f"{tag}  (threshold {thr:.2f})", fontsize=9)
    fig.tight_layout()
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("saved", args.out)


if __name__ == "__main__":
    main()
