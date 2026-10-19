"""Bundle one fold's best checkpoint for the Gradio app.

The bundle holds the weights, class names, preprocessing constants, the
temperature fitted on that fold's validation set, and the clinical class lists
(blast / non-nucleated) agreed by the team.

Usage:
    python scripts/export_model.py --run runs/convnext_ce/proxy --fold 0 \
        --blast-classes <name> [...] --non-nucleated <name> [...] \
        --out app/bonescope_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mds.calibration import fit_temperature
from mds.evaluate import SPLIT_CAVEAT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--blast-classes", nargs="+", required=True)
    parser.add_argument("--non-nucleated", nargs="*", default=[])
    parser.add_argument("--out", default="app/bonescope_model.pt", type=Path)
    args = parser.parse_args()

    fold_dir = args.run / f"fold{args.fold}"
    ckpt = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
    unknown = set(args.blast_classes + args.non_nucleated) - set(ckpt["classes"])
    if unknown:
        raise SystemExit(f"Not model classes: {sorted(unknown)}")
    val = np.load(fold_dir / "val_preds.npz", allow_pickle=True)
    temperature = fit_temperature(val["logits"], val["labels"])
    split_type = ckpt["config"]["data"]["split_type"]
    bundle = {
        **ckpt,
        "temperature": temperature,
        "blast_classes": args.blast_classes,
        "non_nucleated_classes": args.non_nucleated,
        "source": f"{args.run} fold {args.fold}",
        "split_caveat": SPLIT_CAVEAT.get(split_type, split_type),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size:.1f} MB), temperature {temperature:.3f}")


if __name__ == "__main__":
    main()
