"""Train one outer fold: linear probe, then full fine-tune; select on val macro-F1.

Usage:
    python -m mds.train --config configs/convnext_tiny.yaml --fold 0
    python -m mds.train --config configs/convnext_tiny.yaml --fold 0 \
        --set data.root=/content/data/raw train.epochs=3

Writes runs/<run_name>/<split_type>/fold<k>/:
    config.yaml        the exact resolved config
    classes.json       class names in label order
    log.csv            one row per epoch
    best.pt            weights with the best validation macro-F1
    val_preds.npz      validation logits (for temperature scaling)
    test_preds.npz     test-fold logits, labels and paths (for every later analysis)
    metrics.json       test metrics of the best checkpoint
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from mds import DISCLAIMER
from mds.config import Config, load_config, save_config
from mds.data import MarrowDataset, ReinhardStainNormalizer, build_transforms, sqrt_inverse_sampler
from mds.evaluate import SPLIT_CAVEAT, compute_metrics
from mds.losses import build_loss
from mds.models import (
    classifier_parameters,
    count_trainable,
    create_model,
    model_mean_std,
    param_groups,
    set_backbone_trainable,
    set_probe_mode,
)
from mds.splits import class_list, label_counts, load_splits, partition
from mds.utils import (
    CSVLogger,
    amp_enabled,
    get_device,
    get_logger,
    load_json,
    save_json,
    seed_everything,
)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    import random

    random.seed(seed)


def build_loaders(
    cfg: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    d = cfg.data
    stain = None
    if d.stain_norm:
        from PIL import Image

        stain = ReinhardStainNormalizer(Image.open(Path(d.root) / d.stain_ref))
    tf_train = build_transforms(d.img_size, True, mean, std, d.resize_mode, d.canvas_px)
    tf_eval = build_transforms(d.img_size, False, mean, std, d.resize_mode, d.canvas_px)
    common = {
        "num_workers": d.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": d.num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    gen = torch.Generator().manual_seed(cfg.seed)
    sampler = (
        sqrt_inverse_sampler(train_df["label"], cfg.seed) if d.sampler == "sqrt_inverse" else None
    )
    train_dl = DataLoader(
        MarrowDataset(train_df, d.root, tf_train, stain),
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=len(train_df) > cfg.train.batch_size,
        generator=gen,
        **common,
    )
    eval_kw = {"batch_size": cfg.train.batch_size * 2, "shuffle": False, **common}
    val_dl = DataLoader(MarrowDataset(val_df, d.root, tf_eval, stain), **eval_kw)
    test_dl = DataLoader(MarrowDataset(test_df, d.root, tf_eval, stain), **eval_kw)
    return train_dl, val_dl, test_dl


def warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay to `min_lr`, applied to every group's own LR.

    Warmup: right after unfreezing, gradients from the still-noisy head can be
    large; ramping the LR up over the first epoch avoids damaging the backbone.
    """
    lambdas = []
    for group in optimizer.param_groups:
        floor = min(min_lr / group["lr"], 1.0)

        def f(step: int, floor: float = floor) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        lambdas.append(f)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    probe: bool,
) -> float:
    if probe:
        set_probe_mode(model)
    else:
        model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
        with ctx:
            loss = criterion(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], grad_clip
            )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels) for a loader, in dataset order."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
        with ctx:
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def evaluate_loader(
    model, loader, criterion, device, use_amp
) -> tuple[dict, np.ndarray, np.ndarray]:
    logits, labels = predict(model, loader, device, use_amp)
    metrics = compute_metrics(labels, _softmax(logits))
    with torch.no_grad():
        metrics["loss"] = float(
            criterion(torch.as_tensor(logits).to(device), torch.as_tensor(labels).to(device))
        )
    return metrics, logits, labels


def run_dir_for(cfg: Config, config_path: str | Path, fold: int) -> Path:
    name = cfg.output.run_name or Path(config_path).stem
    return Path(cfg.output.dir) / name / cfg.data.split_type / f"fold{fold}"


def train_fold(cfg: Config, fold: int, config_path: str | Path = "config") -> dict:
    """Train and evaluate one outer fold. Returns the test metrics."""
    seed_everything(cfg.seed + fold)
    out = run_dir_for(cfg, config_path, fold)
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"mds.fold{fold}.{out}", out / "train.log")
    log.info(DISCLAIMER)

    # --- data + leakage check (partition raises LeakageError on any overlap) ---
    splits, inner = load_splits(cfg.data.splits_csv, cfg.data.split_type)
    classes = class_list(splits)
    train_df, val_df, test_df = partition(splits, inner, cfg.data.split_type, fold)
    log.info(
        "Leakage check passed [%s]: train %d imgs/%d groups, val %d/%d, test %d/%d",
        cfg.data.split_type, len(train_df), train_df.group.nunique(),
        len(val_df), val_df.group.nunique(), len(test_df), test_df.group.nunique(),
    )  # fmt: skip
    save_config(cfg, out / "config.yaml")
    save_json(classes, out / "classes.json")

    device = get_device(cfg.train.device)
    use_amp = amp_enabled(device, cfg.train.amp)
    log.info("device=%s amp=%s classes=%d", device, use_amp, len(classes))

    model = create_model(cfg, num_classes=len(classes)).to(device)
    mean, std = model_mean_std(model)
    train_dl, val_dl, test_dl = build_loaders(cfg, train_df, val_df, test_df, mean, std, device)
    counts = label_counts(train_df["label"], len(classes))
    criterion = build_loss(cfg.loss, counts).to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)  # same val loss across experiments
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    csv_log = CSVLogger(out / "log.csv")

    wandb_run = None
    if cfg.output.wandb:
        import wandb

        wandb_run = wandb.init(
            project="bonescope", name=f"{out.parent.parent.name}-fold{fold}", config=cfg.to_dict()
        )

    best = {"val_macro_f1": -1.0, "epoch": -1, "stage": ""}
    epoch_idx = 0

    def end_epoch(stage: str, train_loss: float, optimizer) -> bool:
        """Validate, log, keep the best checkpoint. Returns True if improved."""
        nonlocal epoch_idx
        vm, _, _ = evaluate_loader(model, val_dl, eval_criterion, device, use_amp)
        lrs = {g.get("name", f"g{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)}
        row = {
            "epoch": epoch_idx,
            "stage": stage,
            "train_loss": round(train_loss, 5),
            "val_loss": round(vm["loss"], 5),
            "val_macro_f1": round(vm["macro_f1"], 5),
            "val_balanced_acc": round(vm["balanced_acc"], 5),
            "val_top3": round(vm["top3_acc"], 5),
            "val_acc": round(vm["accuracy"], 5),
            "lr_head": lrs.get("head", next(iter(lrs.values()))),
            "time_s": round(time.time() - t0, 1),
        }
        csv_log.log(row)
        if wandb_run is not None:
            wandb_run.log(row)
        improved = vm["macro_f1"] > best["val_macro_f1"]
        log.info(
            "[%s] epoch %d train_loss %.4f val_loss %.4f val_macroF1 %.4f bal_acc %.4f top3 %.4f%s",
            stage, epoch_idx, train_loss, vm["loss"], vm["macro_f1"], vm["balanced_acc"],
            vm["top3_acc"], "  *best*" if improved else "",
        )  # fmt: skip
        if improved:
            best.update(val_macro_f1=vm["macro_f1"], epoch=epoch_idx, stage=stage)
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "config": cfg.to_dict(),
                    "mean": mean,
                    "std": std,
                    "epoch": epoch_idx,
                    "val_macro_f1": vm["macro_f1"],
                },
                out / "best.pt",
            )
        epoch_idx += 1
        return improved

    # --- stage 1: linear probe (frozen backbone, head only) ---
    t0 = time.time()
    set_backbone_trainable(model, False)
    log.info("Stage 1 (linear probe): %d trainable params", count_trainable(model))
    opt = torch.optim.AdamW(
        [{"name": "head", "params": classifier_parameters(model), "lr": cfg.train.probe_lr}],
        weight_decay=cfg.train.weight_decay,
    )
    probe_only = cfg.model.linear_probe_only
    stale = 0
    for _ in range(cfg.train.probe_epochs):
        loss = train_one_epoch(
            model, train_dl, criterion, opt, None, scaler, device, use_amp,
            cfg.train.grad_clip, probe=True,
        )  # fmt: skip
        stale = 0 if end_epoch("probe", loss, opt) else stale + 1
        if probe_only and stale >= cfg.train.patience:
            log.info("Early stopping (probe): no val macro-F1 gain for %d epochs", stale)
            break

    # --- stage 2: full fine-tune with discriminative LRs ---
    if not probe_only and cfg.train.epochs > 0:
        set_backbone_trainable(model, True)
        groups = param_groups(model, cfg.train)
        log.info(
            "Stage 2 (fine-tune): %d trainable params; groups %s",
            count_trainable(model), {g["name"]: g["lr"] for g in groups},
        )  # fmt: skip
        opt = torch.optim.AdamW(groups)
        steps = len(train_dl)
        sched = warmup_cosine(
            opt, steps * cfg.train.epochs, steps * cfg.train.warmup_epochs, cfg.train.min_lr
        )
        stale = 0
        for _ in range(cfg.train.epochs):
            loss = train_one_epoch(
                model, train_dl, criterion, opt, sched, scaler, device, use_amp,
                cfg.train.grad_clip, probe=False,
            )  # fmt: skip
            stale = 0 if end_epoch("finetune", loss, opt) else stale + 1
            if stale >= cfg.train.patience:
                log.info("Early stopping: no val macro-F1 gain for %d epochs", stale)
                break

    # --- final evaluation of the best checkpoint ---
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_logits, val_labels = predict(model, val_dl, device, use_amp)
    test_logits, test_labels = predict(model, test_dl, device, use_amp)
    np.savez_compressed(
        out / "val_preds.npz", logits=val_logits, labels=val_labels, paths=val_df["path"].to_numpy()
    )
    np.savez_compressed(
        out / "test_preds.npz",
        logits=test_logits,
        labels=test_labels,
        paths=test_df["path"].to_numpy(),
    )
    metrics = compute_metrics(test_labels, _softmax(test_logits))
    metrics.update(
        fold=fold,
        split_type=cfg.data.split_type,
        split_caveat=SPLIT_CAVEAT[cfg.data.split_type],
        best_epoch=best["epoch"],
        best_stage=best["stage"],
        best_val_macro_f1=best["val_macro_f1"],
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        model=cfg.model.name,
        loss=cfg.loss.name,
        sampler=cfg.data.sampler,
        disclaimer=DISCLAIMER,
    )
    save_json(metrics, out / "metrics.json")
    log.info(
        "TEST fold %d (%s): macro-F1 %.4f | bal-acc %.4f | top-3 %.4f | acc %.4f",
        fold, SPLIT_CAVEAT[cfg.data.split_type], metrics["macro_f1"], metrics["balanced_acc"],
        metrics["top3_acc"], metrics["accuracy"],
    )  # fmt: skip
    if wandb_run is not None:
        wandb_run.summary.update({f"test_{k}": v for k, v in metrics.items()})
        wandb_run.finish()
    return metrics


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Train one fold (educational; not for clinical use)"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="section.key=value")
    parser.add_argument(
        "--skip-done", action="store_true", help="skip if this fold already has metrics.json"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    done = run_dir_for(cfg, args.config, args.fold) / "metrics.json"
    if args.skip_done and done.exists():
        get_logger("mds").info("Skipping fold %d: %s exists", args.fold, done)
        return load_json(done)
    return train_fold(cfg, args.fold, args.config)


if __name__ == "__main__":
    main()
