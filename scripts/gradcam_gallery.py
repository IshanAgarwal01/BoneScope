"""Grad-CAM gallery: per class, 2 correct and 2 incorrect test cells of one fold.

Each tile shows the heat map for the PREDICTED class, plus its "border heat"
(share of heat in the outer 15% frame). Tiles with border heat > 0.6 are
marked as possible shortcuts: the model may be looking at background or a
neighbouring cell rather than the cell itself.

Usage:
    python scripts/gradcam_gallery.py --run runs/convnext_tiny/proxy --fold 0 \
        [--root data/raw] [--out reports]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from mds.data import build_transforms  # noqa: E402
from mds.explain import CamExplainer, border_heat_fraction, denormalize, overlay  # noqa: E402
from mds.models import load_checkpoint  # noqa: E402

SHORTCUT_THRESHOLD = 0.6


def pick_examples(labels, preds, n_classes, per_kind=2, seed=0) -> list[tuple[int, int, str]]:
    """(class, index, 'correct'|'wrong') picks: up to `per_kind` each per true class."""
    rng = np.random.default_rng(seed)
    picks = []
    for c in range(n_classes):
        for kind, mask in (("correct", preds == labels), ("wrong", preds != labels)):
            idx = np.flatnonzero((labels == c) & mask)
            for i in rng.choice(idx, size=min(per_kind, len(idx)), replace=False):
                picks.append((c, int(i), kind))
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--root", type=Path, help="defaults to data.root in the run config")
    parser.add_argument("--out", default="reports", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    fold_dir = args.run / f"fold{args.fold}"
    model, ckpt = load_checkpoint(str(fold_dir / "best.pt"))
    classes = ckpt["classes"]
    cfg = ckpt["config"]
    root = args.root or Path(cfg["data"]["root"])
    d = cfg["data"]
    tf = build_transforms(
        d["img_size"], False, ckpt["mean"], ckpt["std"], d["resize_mode"], d["canvas_px"]
    )
    test = np.load(fold_dir / "test_preds.npz", allow_pickle=True)
    labels, preds = test["labels"], test["logits"].argmax(1)
    picks = pick_examples(labels, preds, len(classes), seed=args.seed)

    explainer = CamExplainer(model)
    rows = []
    cols = 4
    n_rows = len(classes)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.1, n_rows * 2.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    slot = {c: 0 for c in range(len(classes))}
    for c, i, kind in picks:
        img = Image.open(root / test["paths"][i]).convert("RGB")
        x = tf(img)
        cam = explainer(x, class_idx=int(preds[i]))[0]
        frac = border_heat_fraction(cam)
        shortcut = bool(frac > SHORTCUT_THRESHOLD)  # NaN (flat map) -> False
        rows.append(
            {
                "path": test["paths"][i],
                "true": classes[labels[i]],
                "pred": classes[preds[i]],
                "kind": kind,
                "border_heat": round(frac, 3),
                "possible_shortcut": shortcut,
            }
        )
        ax = axes[c][slot[c]]
        slot[c] += 1
        ax.imshow(overlay(denormalize(x, ckpt["mean"], ckpt["std"]), cam))
        color = "green" if kind == "correct" else "red"
        flag = " !" if shortcut else ""
        ax.set_title(
            f"T: {classes[labels[i]][:14]}\nP: {classes[preds[i]][:14]}  b={frac:.2f}{flag}",
            fontsize=7,
            color=color,
        )
    explainer.close()
    fig.suptitle(
        f"Grad-CAM (fold {args.fold}; green=correct, red=wrong, b=border heat, "
        f"!=b>{SHORTCUT_THRESHOLD})\nEducational project, not for clinical use",
        fontsize=9,
    )
    fig.tight_layout()
    name = f"{args.run.parent.name}_{args.run.name}_fold{args.fold}"
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "figures" / f"gradcam_{name}.png", dpi=110)
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"gradcam_{name}.csv", index=False)
    print(df.to_string(index=False))
    print(f"\npossible shortcuts: {int(df['possible_shortcut'].sum())} of {len(df)} tiles")


if __name__ == "__main__":
    main()
