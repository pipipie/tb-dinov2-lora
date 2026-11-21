"""Step 3 - turn runs/preds.csv into the tables and figures of the paper.

Usage:  python evaluate.py [--ref resnet50] [--boot 2000]

Outputs (in runs/ and figs/)
  results_main.csv    AUC / sensitivity / specificity / F1 with 95% bootstrap CIs, every setting
  results_gap.csv     in-domain (5-fold CV) minus cross-dataset AUC, with CI  <- main claim
  results_paired.csv  AUC difference of each model vs. a reference model on the same test set
  figs/roc_cross_*.png

How seeds and CIs are handled
  * Point estimate = mean over seeds of the metric.
  * 95% CI = percentile bootstrap over TEST IMAGES (stratified by class, 2000 resamples);
    inside each resample the metric is computed per seed and then averaged over seeds.
  * seed_sd = standard deviation of the metric across seeds (training randomness only).
  * Sensitivity / specificity / F1 use the threshold chosen on the validation split.
"""
import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(os.environ.get("TB_ROOT", "."))  # same convention as common.py, but no torch needed here
RUNS_DIR, FIG_DIR = ROOT / "runs", ROOT / "figs"

METRICS = ["auc", "sensitivity", "specificity", "f1"]
KEY = ["protocol", "tag", "train_source", "test_source"]


# ----------------------------------------------------------------------------- metrics
def binary_metrics(y, prob, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return np.array([
        roc_auc_score(y, prob),
        tp / max(tp + fn, 1),
        tn / max(tn + fp, 1),
        2 * tp / max(2 * tp + fp + fn, 1),
    ])


def seed_avg(y, P, PR, idx):
    """Mean over seeds of [auc, sens, spec, f1] on the images `idx`."""
    return np.mean([binary_metrics(y[idx], P[s, idx], PR[s, idx]) for s in range(P.shape[0])], axis=0)


def boot_indices(y, B, seed=0):
    """Stratified bootstrap indices. Depends only on y and seed, so two models evaluated on the
    same test set get IDENTICAL resamples -> valid paired comparisons."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(B)]


# ----------------------------------------------------------------------------- data
def build_groups(df):
    groups = {}
    for key, sub in df.groupby(KEY):
        n_full = sub.groupby("seed").size().max()
        mats_p, mats_r, ids, y = [], [], None, None
        for seed, ss in sub.groupby("seed"):
            if len(ss) != n_full:
                print(f"WARNING {key}: seed {seed} incomplete ({len(ss)}/{n_full} images) - skipped")
                continue
            ss = ss.sort_values("id")
            if ss["id"].duplicated().any():
                raise ValueError(f"{key}: duplicated image ids within seed {seed}")
            if ids is None:
                ids, y = ss["id"].values, ss["label"].values
            elif not np.array_equal(ids, ss["id"].values):
                raise ValueError(f"{key}: seeds do not cover identical images")
            mats_p.append(ss["prob"].values)
            mats_r.append(ss["pred"].values)
        if mats_p:
            groups[key] = dict(y=y, P=np.array(mats_p), PR=np.array(mats_r), ids=ids)
    return groups


def fmt(est, lo, hi):
    return f"{est:.3f} [{lo:.3f}-{hi:.3f}]"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(RUNS_DIR / "preds.csv"))
    ap.add_argument("--ref", default="resnet50", help="reference model for paired comparisons")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    groups = build_groups(df)
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    boots, summary, wide = {}, [], []
    for key, g in groups.items():
        y, P, PR = g["y"], g["P"], g["PR"]
        idxs = boot_indices(y, args.boot)
        arr = np.array([seed_avg(y, P, PR, i) for i in idxs])            # (B, 4)
        est = seed_avg(y, P, PR, np.arange(len(y)))
        per_seed = np.array([binary_metrics(y, P[s], PR[s]) for s in range(P.shape[0])])
        sd = per_seed.std(axis=0, ddof=1) if P.shape[0] > 1 else np.full(4, np.nan)
        lo, hi = np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
        boots[key] = arr
        row = dict(zip(KEY, key), n_images=len(y), n_pos=int(y.sum()), n_seeds=P.shape[0])
        for j, m in enumerate(METRICS):
            summary.append(dict(zip(KEY, key), metric=m, estimate=est[j], ci_low=lo[j], ci_high=hi[j], seed_sd=sd[j]))
            row[m] = fmt(est[j], lo[j], hi[j])
        wide.append(row)

    main_tbl = pd.DataFrame(wide).sort_values(["protocol", "train_source", "tag"])
    main_tbl.to_csv(RUNS_DIR / "results_main.csv", index=False)
    pd.DataFrame(summary).to_csv(RUNS_DIR / "results_main_long.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 30)
    print("\n=== MAIN RESULTS (mean over seeds, 95% bootstrap CI) ===")
    print(main_tbl.to_string(index=False))

    # ---- generalization gap: in-domain (cv) AUC minus cross-dataset AUC (independent test sets)
    gap_rows = []
    for (proto, tag, src, tgt), arr_x in boots.items():
        if proto != "cross" or ("cv", tag, src, src) not in boots:
            continue
        arr_in = boots[("cv", tag, src, src)]
        d = arr_in[:, 0] - arr_x[:, 0]
        gap_rows.append(dict(tag=tag, train_source=src, cross_test_source=tgt,
                             auc_in_domain=arr_in[:, 0].mean(), auc_cross=arr_x[:, 0].mean(),
                             auc_gap=d.mean(), gap_ci_low=np.percentile(d, 2.5), gap_ci_high=np.percentile(d, 97.5)))
    gap = pd.DataFrame(gap_rows).sort_values(["train_source", "tag"]) if gap_rows else pd.DataFrame()
    gap.to_csv(RUNS_DIR / "results_gap.csv", index=False)
    print("\n=== GENERALIZATION GAP (AUC in-domain minus cross-dataset; smaller = better) ===")
    print(gap.round(3).to_string(index=False) if len(gap) else "(need both 'cv' and 'cross' runs)")

    # ---- paired comparison vs reference model, same test set, same bootstrap resamples
    paired = []
    for (proto, tag, src, tgt), arr in boots.items():
        ref_key = (proto, args.ref, src, tgt)
        if tag == args.ref or ref_key not in boots:
            continue
        d = arr[:, 0] - boots[ref_key][:, 0]
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        paired.append(dict(protocol=proto, train_source=src, test_source=tgt, model=tag, reference=args.ref,
                           delta_auc=d.mean(), ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5), p_boot=p))
    paired = pd.DataFrame(paired)
    paired.to_csv(RUNS_DIR / "results_paired.csv", index=False)
    print(f"\n=== PAIRED AUC DIFFERENCE vs {args.ref} (positive = better than reference) ===")
    print(paired.round(3).to_string(index=False) if len(paired) else "(no pairs yet)")
    print("p_boot = approximate two-sided bootstrap p-value; apply a multiple-comparison correction "
          "(e.g. Holm) across the models you compare and report it as exploratory.")

    # ---- ablation view (LoRA rank / linear probe) on cross-dataset tests
    abl = main_tbl[(main_tbl.protocol == "cross") & main_tbl.tag.str.startswith("dinov2")]
    if len(abl):
        print("\n=== ABLATION: linear probe vs LoRA rank (cross-dataset AUC) ===")
        print(abl[["train_source", "test_source", "tag", "auc", "sensitivity", "specificity"]].to_string(index=False))

    # ---- ROC curves for cross-dataset tests (first seed of each model)
    for (src, tgt), _ in main_tbl[main_tbl.protocol == "cross"].groupby(["train_source", "test_source"]):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for key, g in groups.items():
            if key[0] == "cross" and key[2] == src and key[3] == tgt:
                fpr, tpr, _ = roc_curve(g["y"], g["P"][0])
                ax.plot(fpr, tpr, label=f"{key[1]} (AUC {roc_auc_score(g['y'], g['P'][0]):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="1 - specificity", ylabel="sensitivity", title=f"Train {src} -> test {tgt}\n(seed {int(df.seed.min())})")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"roc_cross_{src}_to_{tgt}.png", dpi=200)
        plt.close(fig)
    print(f"\nSaved tables to {RUNS_DIR}/ and ROC figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
