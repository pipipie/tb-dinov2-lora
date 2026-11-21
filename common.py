"""Shared helpers for the TB cross-hospital generalization project.

Everything that must be identical between training, evaluation, Grad-CAM, the RSNA
check and the web demo lives here (preprocessing, model construction, checkpoints).
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------- paths
ROOT = Path(os.environ.get("TB_ROOT", "."))
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RUNS_DIR = ROOT / "runs"
CKPT_DIR = ROOT / "ckpt"
FIG_DIR = ROOT / "figs"
MANIFEST = ROOT / "manifest.csv"

# ----------------------------------------------------------------------------- config
IMG_SIZE = 224  # 224 = 16 x 14, a multiple of DINOv2's patch size
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_ID = "facebook/dinov2-base"
# Set TB_NO_PRETRAINED=1 only for offline smoke tests (tiny random models, no downloads).
PRETRAINED = os.environ.get("TB_NO_PRETRAINED", "0") != "1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------- images
def load_gray8(path) -> Image.Image:
    """Open an image and return an 8-bit grayscale PIL image (handles 16-bit PNGs)."""
    im = Image.open(path)
    if im.mode not in ("L", "RGB", "RGBA"):
        a = np.asarray(im, dtype=np.float32)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255.0
        im = Image.fromarray(a.astype(np.uint8))
    return im.convert("L")


def preprocess_pil(im: Image.Image) -> Image.Image:
    """The single preprocessing step used everywhere: gray -> 224x224 -> 3 channels."""
    return im.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS).convert("RGB")


TRAIN_TF = T.Compose([
    # No horizontal flip: heart/stomach-bubble side carries anatomical meaning.
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
EVAL_TF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class CXRDataset(Dataset):
    """Reads the cached 224x224 PNGs listed in the manifest."""

    def __init__(self, df: pd.DataFrame, train: bool = False):
        self.paths = df["cache_path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tf = TRAIN_TF if train else EVAL_TF

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def make_loader(df, train=False, bs=32, num_workers=2, seed=0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        CXRDataset(df, train), batch_size=bs, shuffle=train, drop_last=train,
        num_workers=num_workers, pin_memory=DEVICE == "cuda", generator=g if train else None,
    )


# ----------------------------------------------------------------------------- models
def make_tag(model: str, rank: int | None = None) -> str:
    return f"dinov2_lora_r{rank}" if model == "dinov2_lora" else model


def parse_tag(tag: str):
    m = re.fullmatch(r"dinov2_lora_r(\d+)", tag)
    return ("dinov2_lora", int(m.group(1))) if m else (tag, None)


class DinoClassifier(nn.Module):
    """DINOv2 backbone + linear head on [CLS ; mean(patch tokens)].

    mode="linear": backbone frozen (linear probe).
    mode="lora"  : backbone frozen except LoRA adapters on the attention query/value.
    """

    def __init__(self, mode: str = "lora", rank: int = 8, pretrained: bool = True):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model

        if pretrained:
            backbone = Dinov2Model.from_pretrained(DINO_ID)
        else:  # tiny random model for offline smoke tests only
            backbone = Dinov2Model(Dinov2Config(hidden_size=64, num_hidden_layers=2,
                                                num_attention_heads=2, image_size=IMG_SIZE,
                                                patch_size=14))
        hidden = backbone.config.hidden_size
        for p in backbone.parameters():
            p.requires_grad = False
        if mode == "lora":
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.1,
                             target_modules=["query", "value"], bias="none")
            backbone = get_peft_model(backbone, cfg)  # only LoRA weights stay trainable
        self.backbone = backbone
        self.head = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        feat = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
        return self.head(feat)


def build_model(tag: str, pretrained: bool | None = None) -> nn.Module:
    pretrained = PRETRAINED if pretrained is None else pretrained
    name, rank = parse_tag(tag)
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    if name == "densenet121":
        from torchvision.models import DenseNet121_Weights, densenet121

        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        return m
    if name == "dinov2_linear":
        return DinoClassifier("linear", pretrained=pretrained)
    if name == "dinov2_lora":
        return DinoClassifier("lora", rank=rank, pretrained=pretrained)
    raise ValueError(f"unknown model tag: {tag}")


def is_dino(tag: str) -> bool:
    return tag.startswith("dinov2")


def snapshot(model: nn.Module, tag: str) -> dict:
    """State to keep for 'best epoch' / checkpoints.

    DINOv2 models: only trainable tensors (LoRA + head) -> tiny files.
    CNNs: the full state dict (BatchNorm running statistics matter).
    """
    sd = model.state_dict()
    if is_dino(tag):
        keep = {n for n, p in model.named_parameters() if p.requires_grad}
        sd = {k: v for k, v in sd.items() if k in keep}
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def restore(model: nn.Module, sd: dict, tag: str) -> None:
    model.load_state_dict(sd, strict=not is_dino(tag))


def save_ckpt(path, model, tag: str, thr: float, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tag": tag, "thr": float(thr), "state": snapshot(model, tag), **(extra or {})}, path)


def load_ckpt(path, device: str = DEVICE):
    ck = torch.load(path, map_location="cpu")
    tag = ck["tag"]
    # CNN weights are fully stored in the checkpoint; DINOv2's frozen backbone is re-downloaded.
    model = build_model(tag, pretrained=(PRETRAINED if is_dino(tag) else False))
    restore(model, ck["state"], tag)
    return model.to(device).eval(), tag, float(ck["thr"])


# ----------------------------------------------------------------------------- inference / metrics
@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: str = DEVICE) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(x.to(device))
        out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
    return torch.cat(out).numpy()


def youden_threshold(y, p) -> float:
    """Threshold maximising sensitivity + specificity - 1 (chosen on validation data only)."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.where(np.isfinite(thr), thr, 1.0)
    return float(thr[np.argmax(tpr - fpr)])
