# Results

**No experiments have been run on the real dataset yet.**

This file is overwritten by `python scripts/aggregate_results.py` once runs exist.
Each row will show mean ± std over 5 folds and state its split type; proxy-group and
image-level splits are labelled *"may be optimistic"*.

Planned experiments (Phase 4, one change at a time, same frozen folds):

| # | run_name | change from baseline |
|---|---|---|
| 1 | `convnext_ce` | baseline: ConvNeXt-Tiny, CE + label smoothing 0.1 |
| 2 | `convnext_cbce` | class-balanced CE (beta 0.999) |
| 3 | `convnext_focal` | focal loss (gamma 2) instead |
| 4 | `convnext_sqrt` | sqrt-inverse sampler instead of a weighted loss |
| 5 | `effnet_best` | best of 1-4 on EfficientNet-B3 |
| 6 | `dinov2_probe` | frozen DINOv2 ViT-S/14 + linear head |
| 7 | best config | full 5-fold CV on proxy **and** image-level splits |
