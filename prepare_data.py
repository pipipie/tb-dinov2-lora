"""Step 1 - download Montgomery + Shenzhen, parse labels, cache 224x224 images, make CV folds.

Usage:  python prepare_data.py
Output: manifest.csv  (columns: name, source, label, path, cache_path, fold)
        cache/<source>/<name>.png
"""
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from common import CACHE_DIR, DATA_DIR, MANIFEST, load_gray8, preprocess_pil

REPO = "Famatsu123/montgomery-shenzhen-tuberculosis-cxr"
SOURCES = {  # source -> folder inside the repo
    "shenzhen": "ChinaSet_AllFiles/CXR_png",
    "montgomery": "MontgomerySet/CXR_png",
}
# Counts from the original NLM release (Jaeger et al., 2014): total, normal, TB
EXPECTED = {"shenzhen": (662, 326, 336), "montgomery": (138, 80, 58)}
N_FOLDS = 5
FOLD_SEED = 42


def main():
    # Only the chest X-ray PNGs are needed (skips masks / clinical readings).
    local = Path(snapshot_download(
        repo_id=REPO, repo_type="dataset", local_dir=DATA_DIR / "hf",
        allow_patterns=[f"{folder}/*.png" for folder in SOURCES.values()],
    ))

    rows = []
    for source, folder in SOURCES.items():
        files = sorted((local / folder).glob("*.png"))
        print(f"{source}: found {len(files)} PNG files")
        out_dir = CACHE_DIR / source
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in tqdm(files, desc=f"caching {source}"):
            # Filename convention: CHNCXR_0001_1.png / MCUCXR_0001_0.png -> last token = label
            label = int(f.stem.split("_")[-1])
            assert label in (0, 1), f"unexpected label in {f.name}"
            dst = out_dir / f"{f.stem}.png"
            if not dst.exists():
                preprocess_pil(load_gray8(f)).save(dst)
            rows.append(dict(name=f.stem, source=source, label=label, path=str(f), cache_path=str(dst)))

    df = pd.DataFrame(rows)

    # Stratified 5-fold CV inside each source (one image per patient in these collections,
    # so image-level splitting does not leak patients - verify this claim in your paper).
    df["fold"] = -1
    for source, sub in df.groupby("source"):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
        for k, (_, test_idx) in enumerate(skf.split(sub, sub["label"])):
            df.loc[sub.index[test_idx], "fold"] = k

    df.to_csv(MANIFEST, index=False)

    # ---- sanity checks against the published counts
    print("\nSource      total  normal  TB")
    for source, (n, n0, n1) in EXPECTED.items():
        sub = df[df.source == source]
        got = (len(sub), int((sub.label == 0).sum()), int((sub.label == 1).sum()))
        flag = "OK" if got == (n, n0, n1) else f"MISMATCH (expected {n}/{n0}/{n1})"
        print(f"{source:<11} {got[0]:>5} {got[1]:>7} {got[2]:>4}   {flag}")
    print(f"\nWrote {MANIFEST} with {len(df)} images")


if __name__ == "__main__":
    main()
