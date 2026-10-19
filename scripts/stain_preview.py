"""Show 20 crops before/after Reinhard stain normalisation.

Look at this figure BEFORE enabling data.stain_norm: if blast cytoplasm stops
looking blue or eosinophil granules stop looking orange, normalisation is
erasing signal, not just noise.

Usage:
    python scripts/stain_preview.py --root data/raw --ref <relative path of a typical crop>
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from mds.data import ReinhardStainNormalizer, index_images  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/raw", type=Path)
    parser.add_argument("--ref", required=True, help="reference crop, relative to --root")
    parser.add_argument("--n", default=20, type=int)
    parser.add_argument("--out", default="reports/figures/stain_preview.png", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    df = index_images(args.root)
    norm = ReinhardStainNormalizer(Image.open(args.root / args.ref))
    picks = df.sample(n=min(args.n, len(df)), random_state=random.Random(args.seed).randint(0, 999))
    cols = 5
    rows = 2 * ((len(picks) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (_, r) in enumerate(picks.iterrows()):
        img = Image.open(args.root / r["path"]).convert("RGB")
        top, col = 2 * (i // cols), i % cols
        axes[top][col].imshow(img)
        axes[top][col].set_title(f"{r['label_name']}\noriginal", fontsize=7)
        axes[top + 1][col].imshow(norm(img))
        axes[top + 1][col].set_title("Reinhard", fontsize=7)
    fig.suptitle(f"Stain normalisation preview (reference: {args.ref})")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
