"""Build proxy "pseudo-patient" groups when the release has no patient/slide IDs.

Each image gets (a) a frozen pretrained CNN embedding and (b) the colour of its
background border. K-means on the combined features gives ~125 clusters, which
stand in for patients when building StratifiedGroupKFold folds.

Why this helps: images from the same slide share stain batch, smear thickness
and microscope lighting, so they tend to land in the same cluster; holding a
cluster out forces the model to generalise to unseen imaging conditions.
Why it is only approximate: a patient's cells can land in several clusters, so
some patient-level leakage can remain. Results are labelled accordingly.

Usage:
    python scripts/compute_proxy_groups.py --root data/raw --out splits/proxy_groups.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from tqdm import tqdm

from mds.data import (
    border_color_stats,
    build_proxy_features,
    cluster_class_purity,
    cluster_proxy_groups,
    index_images,
)
from mds.utils import get_device, seed_everything


@torch.no_grad()
def embed_images(
    root: Path,
    paths: list[str],
    model_name: str,
    pretrained: bool,
    batch_size: int = 64,
    img_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pooled CNN embeddings, border colour stats) for every image."""
    device = get_device("auto")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0).eval().to(device)
    data_cfg = timm.data.resolve_data_config({"input_size": (3, img_size, img_size)}, model=model)
    tf = timm.data.create_transform(**data_cfg, is_training=False)
    embs, colors = [], []
    for i in tqdm(range(0, len(paths), batch_size), desc="embedding"):
        batch, batch_colors = [], []
        for rel in paths[i : i + batch_size]:
            with Image.open(root / rel) as img:
                img = img.convert("RGB")
                batch_colors.append(border_color_stats(img))
                batch.append(tf(img))
        embs.append(model(torch.stack(batch).to(device)).float().cpu().numpy())
        colors.extend(batch_colors)
    return np.concatenate(embs), np.stack(colors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/raw", type=Path)
    parser.add_argument("--out", default="splits/proxy_groups.csv", type=Path)
    parser.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument("--no-pretrained", action="store_true", help="tests only")
    parser.add_argument("--n-groups", default=125, type=int, help="dataset has 125 patients")
    parser.add_argument("--color-weight", default=2.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    seed_everything(args.seed)
    df = index_images(args.root)
    emb, col = embed_images(args.root, df["path"].tolist(), args.model, not args.no_pretrained)
    feats = build_proxy_features(emb, col, color_weight=args.color_weight, seed=args.seed)
    df["group"] = cluster_proxy_groups(feats, args.n_groups, seed=args.seed)

    purity = cluster_class_purity(df["group"].to_numpy(), df["label_name"].to_numpy())
    chance = df["label_name"].value_counts(normalize=True).iloc[0]
    sizes = df["group"].value_counts()
    print(
        f"clusters: {df['group'].nunique()}  size min/median/max: "
        f"{sizes.min()}/{int(sizes.median())}/{sizes.max()}"
    )
    print(
        f"cluster class purity: {purity:.3f} (largest-class share = {chance:.3f}). "
        "Purity near 1.0 means clusters follow cell type, not slide; "
        "raise --color-weight and re-run."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["path", "label_name", "group"]].to_csv(args.out, index=False)
    np.save(args.out.with_suffix(".features.npy"), feats)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
