"""Step 2 - train models and write predictions.

Protocols
  cross : train on one source, test on the OTHER source      (the generalization test)
  cv    : stratified 5-fold CV inside each source             (the in-domain reference)
  final : train on both sources, save checkpoint for the demo (no evaluation)

Every run: 80/20 train/val split of the training data -> early stopping on val AUC ->
decision threshold chosen on the val split (Youden) -> applied unchanged to the test data.
The test data is never used for model selection or thresholding.

Examples
  python train.py --protocol cross --models resnet50 densenet121 dinov2_linear dinov2_lora --ranks 4 8 16 --seeds 0 1 2 --save_ckpt
  python train.py --protocol cv    --models resnet50 densenet121 dinov2_linear dinov2_lora --ranks 4 8 16 --seeds 0
  python train.py --protocol final --models resnet50 dinov2_lora --ranks 8

Predictions are appended to runs/preds.csv; finished runs are skipped, so you can re-run
the same command after a Colab disconnect.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import (CKPT_DIR, DEVICE, MANIFEST, RUNS_DIR, build_model, make_loader, make_tag,
                    predict_proba, restore, save_ckpt, set_seed, snapshot, youden_threshold)

PREDS = RUNS_DIR / "preds.csv"
SOURCES = ["shenzhen", "montgomery"]


def default_lr(tag: str) -> float:
    if tag.startswith("dinov2_lora"):
        return 3e-4
    if tag == "dinov2_linear":
        return 1e-3
    return 1e-4  # full fine-tuning of the CNN baselines


def fit_predict(tag, train_df, test_sets, seed, epochs=30, bs=32, lr=None, patience=7,
                ckpt_path=None, num_workers=2):
    """Train on train_df, return threshold + probabilities for every df in test_sets."""
    lr = lr or default_lr(tag)
    set_seed(seed)
    tr, va = train_test_split(train_df, test_size=0.2, stratify=train_df["label"], random_state=seed)

    model = build_model(tag).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {tag}: {n_train/1e6:.2f}M trainable / {n_total/1e6:.1f}M total parameters")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    counts = np.bincount(tr["label"], minlength=2)
    class_w = torch.tensor(len(tr) / (2 * counts), dtype=torch.float32, device=DEVICE)  # class-balanced
    crit = nn.CrossEntropyLoss(weight=class_w)

    train_loader = make_loader(tr, train=True, bs=bs, num_workers=num_workers, seed=seed)
    val_loader = make_loader(va, bs=bs, num_workers=num_workers)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        auc = roc_auc_score(va["label"], predict_proba(model, val_loader))
        if auc > best_auc:
            best_auc, best_state, bad = auc, snapshot(model, tag), 0
        else:
            bad += 1
            if bad >= patience:
                break
    print(f"  stopped after epoch {epoch + 1}, best val AUC {best_auc:.4f}")

    restore(model, best_state, tag)
    thr = youden_threshold(va["label"].values, predict_proba(model, val_loader))
    if ckpt_path is not None:
        save_ckpt(ckpt_path, model, tag, thr, extra={"val_auc": best_auc, "seed": seed})

    probs = {name: predict_proba(model, make_loader(df, bs=bs, num_workers=num_workers))
             for name, df in test_sets.items()}
    del model, opt
    torch.cuda.empty_cache()
    return dict(thr=thr, val_auc=best_auc, probs=probs)


def to_rows(meta, df, probs, thr, val_auc):
    out = pd.DataFrame({"id": df["name"].values, "label": df["label"].values, "prob": probs})
    out["pred"] = (out["prob"] >= thr).astype(int)
    out["thr"], out["val_auc"] = thr, val_auc
    for k, v in meta.items():
        out[k] = v
    return out


def append(rows: pd.DataFrame):
    PREDS.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(PREDS, mode="a", header=not PREDS.exists(), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["cross", "cv", "final"], required=True)
    ap.add_argument("--models", nargs="+", default=["resnet50", "densenet121", "dinov2_linear", "dinov2_lora"],
                    choices=["resnet50", "densenet121", "dinov2_linear", "dinov2_lora"])
    ap.add_argument("--ranks", nargs="+", type=int, default=[8], help="LoRA ranks (used by dinov2_lora)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="override the per-model default")
    ap.add_argument("--save_ckpt", action="store_true", help="cross protocol: keep a checkpoint of the first seed")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    manifest = pd.read_csv(MANIFEST)
    tags = []
    for m in args.models:
        tags += [make_tag(m, r) for r in args.ranks] if m == "dinov2_lora" else [make_tag(m)]

    done = set()
    if PREDS.exists():
        d = pd.read_csv(PREDS, usecols=["protocol", "tag", "seed", "train_source", "test_source", "fold"])
        done = set(map(tuple, d.drop_duplicates().itertuples(index=False, name=None)))

    for tag in tags:
        for seed in args.seeds:
            t0 = time.time()
            if args.protocol == "cross":
                for src in SOURCES:
                    other = [s for s in SOURCES if s != src][0]
                    key = ("cross", tag, seed, src, other, -1)
                    if key in done:
                        continue
                    print(f"[cross] {tag} seed={seed} train={src} -> test={other}")
                    ck = CKPT_DIR / f"{tag}_cross_{src}.pt" if (args.save_ckpt and seed == args.seeds[0]) else None
                    res = fit_predict(tag, manifest[manifest.source == src],
                                      {other: manifest[manifest.source == other]},
                                      seed, args.epochs, args.bs, args.lr, ckpt_path=ck, num_workers=args.workers)
                    meta = dict(protocol="cross", tag=tag, seed=seed, train_source=src, test_source=other, fold=-1)
                    append(to_rows(meta, manifest[manifest.source == other], res["probs"][other], res["thr"], res["val_auc"]))

            elif args.protocol == "cv":
                for src in SOURCES:
                    sub = manifest[manifest.source == src]
                    for k in sorted(sub.fold.unique()):
                        key = ("cv", tag, seed, src, src, int(k))
                        if key in done:
                            continue
                        print(f"[cv] {tag} seed={seed} source={src} fold={k}")
                        test_df = sub[sub.fold == k]
                        res = fit_predict(tag, sub[sub.fold != k], {src: test_df},
                                          seed, args.epochs, args.bs, args.lr, num_workers=args.workers)
                        meta = dict(protocol="cv", tag=tag, seed=seed, train_source=src, test_source=src, fold=int(k))
                        append(to_rows(meta, test_df, res["probs"][src], res["thr"], res["val_auc"]))

            else:  # final: all 800 images, checkpoint for the demo
                ck = CKPT_DIR / f"{tag}_final.pt"
                if ck.exists():
                    continue
                print(f"[final] {tag} seed={seed}")
                fit_predict(tag, manifest, {}, seed, args.epochs, args.bs, args.lr, ckpt_path=ck, num_workers=args.workers)
            print(f"  done in {(time.time() - t0) / 60:.1f} min\n")


if __name__ == "__main__":
    main()
