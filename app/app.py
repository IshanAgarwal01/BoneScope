"""BoneScope Gradio demo: single-cell classification and a batch blast-% demo.

Research demo. Not for clinical use. Trained on Wright-Giemsa single-cell crops.

Model source (first match wins):
  BONESCOPE_MODEL      local path to a bundle from scripts/export_model.py
  BONESCOPE_HF_REPO    Hugging Face model repo id holding the bundle
  BONESCOPE_HF_FILE    file name in that repo (default: bonescope_model.pt)
  ./bonescope_model.pt next to this file
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
# Local checkout: use ../src. On a Hugging Face Space, `mds/` is copied next to app.py.
for candidate in (HERE.parent / "src", HERE):
    if (candidate / "mds").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mds.calibration import softmax_np  # noqa: E402
from mds.clinical import estimate_blast_percentage  # noqa: E402
from mds.data import IMAGE_EXTS, build_transforms  # noqa: E402
from mds.explain import CamExplainer, denormalize, overlay  # noqa: E402
from mds.models import create_model  # noqa: E402

DISCLAIMER = "Research demo. Not for clinical use. Trained on Wright-Giemsa single-cell crops."
LOW_CONFIDENCE = 0.5
MAX_BATCH_IMAGES = 3000
MAX_ZIP_BYTES = 200 * 1024 * 1024


@dataclass
class Bundle:
    model: torch.nn.Module
    classes: list[str]
    transform: object
    mean: tuple
    std: tuple
    temperature: float
    blast_classes: list[str]
    non_nucleated_classes: list[str]
    split_caveat: str
    source: str


def load_bundle(path: str | Path) -> Bundle:
    """Rebuild the model from an exported bundle (no pretrained download needed)."""
    from mds.config import from_dict

    b = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(b["config"])
    cfg.model.pretrained = False
    model = create_model(cfg, num_classes=len(b["classes"]))
    model.load_state_dict(b["model"])
    model.eval()
    d = cfg.data
    tf = build_transforms(d.img_size, False, b["mean"], b["std"], d.resize_mode, d.canvas_px)
    return Bundle(
        model=model,
        classes=list(b["classes"]),
        transform=tf,
        mean=tuple(b["mean"]),
        std=tuple(b["std"]),
        temperature=float(b.get("temperature", 1.0)),
        blast_classes=list(b.get("blast_classes", [])),
        non_nucleated_classes=list(b.get("non_nucleated_classes", [])),
        split_caveat=b.get("split_caveat", "unknown split"),
        source=b.get("source", str(path)),
    )


def resolve_model_path() -> Path | None:
    if os.environ.get("BONESCOPE_MODEL"):
        return Path(os.environ["BONESCOPE_MODEL"])
    if os.environ.get("BONESCOPE_HF_REPO"):
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                os.environ["BONESCOPE_HF_REPO"],
                os.environ.get("BONESCOPE_HF_FILE", "bonescope_model.pt"),
            )
        )
    local = HERE / "bonescope_model.pt"
    return local if local.exists() else None


@torch.no_grad()
def _probs(bundle: Bundle, x: torch.Tensor) -> np.ndarray:
    return softmax_np(bundle.model(x).numpy(), bundle.temperature)


def predict_single(bundle: Bundle, img: Image.Image) -> tuple[dict, np.ndarray, str]:
    """Top-3 calibrated probabilities, Grad-CAM overlay, and a status message."""
    x = bundle.transform(img.convert("RGB"))
    probs = _probs(bundle, x[None])[0]
    top = np.argsort(-probs)[:3]
    top3 = {bundle.classes[i]: float(probs[i]) for i in top}
    explainer = CamExplainer(bundle.model)
    try:
        cam = explainer(x, class_idx=int(top[0]))[0]
    finally:
        explainer.close()
    heat = overlay(denormalize(x, bundle.mean, bundle.std), cam)
    if probs[top[0]] < LOW_CONFIDENCE:
        status = (
            f"⚠️ **Low confidence** (top probability {probs[top[0]]:.0%} < 50%). "
            "The model is unsure; treat all three suggestions as equally tentative."
        )
    else:
        status = f"Top prediction: **{bundle.classes[top[0]]}** ({probs[top[0]]:.0%})."
    return top3, heat, f"{status}\n\n*{DISCLAIMER}*"


def read_zip_images(zip_path: str | Path) -> list[tuple[str, Image.Image]]:
    """Read image members of a zip in memory (nothing is extracted to disk)."""
    if Path(zip_path).stat().st_size > MAX_ZIP_BYTES:
        raise ValueError(f"Zip larger than {MAX_ZIP_BYTES // 2**20} MB")
    images = []
    with zipfile.ZipFile(zip_path) as z:
        members = [
            m
            for m in z.infolist()
            if not m.is_dir()
            and Path(m.filename).suffix.lower() in IMAGE_EXTS
            and not Path(m.filename).name.startswith(".")
            and "__MACOSX" not in m.filename
        ]
        if len(members) > MAX_BATCH_IMAGES:
            raise ValueError(f"Zip has {len(members)} images; the limit is {MAX_BATCH_IMAGES}")
        for m in members:
            with Image.open(io.BytesIO(z.read(m))) as img:
                images.append((m.filename, img.convert("RGB")))
    if not images:
        raise ValueError("No images found in the zip")
    return images


def predict_batch(bundle: Bundle, zip_path: str | Path) -> tuple[pd.DataFrame, str]:
    """Cell-count table and a blast-% summary for a zip of single-cell crops."""
    images = read_zip_images(zip_path)
    probs = []
    for i in range(0, len(images), 64):
        x = torch.stack([bundle.transform(img) for _, img in images[i : i + 64]])
        probs.append(_probs(bundle, x))
    probs = np.concatenate(probs)
    preds = probs.argmax(1)
    counts = np.bincount(preds, minlength=len(bundle.classes))
    table = (
        pd.DataFrame({"class": bundle.classes, "cells": counts})
        .assign(share=lambda d: (d["cells"] / d["cells"].sum()).round(3))
        .sort_values("cells", ascending=False)
    )
    table = table[table["cells"] > 0]

    lines = [f"**{len(images)} cells classified.**"]
    if not bundle.blast_classes:
        lines.append("Blast classes are not configured in this model bundle; no blast % shown.")
    else:
        try:
            for method in ("hard", "soft"):
                est = estimate_blast_percentage(
                    probs, bundle.classes, bundle.blast_classes,
                    bundle.non_nucleated_classes, method=method,
                )  # fmt: skip
                flag = (
                    f" ⚠️ **Borderline**: the 95% CI crosses "
                    f"{', '.join(f'{t:g}%' for t in est.crossed)}."
                    if est.borderline
                    else ""
                )
                lines.append(
                    f"- **{method.title()} count:** blast {est.blast_pct:.1f}% "
                    f"(95% CI {est.ci_low:.1f}-{est.ci_high:.1f}%, "
                    f"{est.n_nucleated:.0f} nucleated cells) -> "
                    f"WHO 2022 **{est.category.who}** (ICC: {est.category.icc}), "
                    f"blast range {est.category.blast_range}.{flag}"
                )
        except ValueError as exc:
            lines.append(f"Blast % unavailable: {exc}")
    if len(images) < 500:
        lines.append(
            f"\nOnly {len(images)} cells: a marrow differential uses at least 500 nucleated "
            "cells, so the interval is wide."
        )
    lines += [
        "",
        "This is only the blast-count axis. Real WHO 2022 classification also needs blood "
        "blasts, Auer rods, cytogenetics and mutations, and ring sideroblasts need an iron "
        "stain this model never saw. Uploaded crops are a convenience sample, not "
        "consecutive cells from a smear.",
        "",
        f"*{DISCLAIMER}*",
    ]
    return table, "\n".join(lines)


def build_demo(bundle: Bundle | None):
    import gradio as gr

    header = (
        "# BoneScope: MDS bone marrow cell classifier\n"
        f"**{DISCLAIMER}**\n\n"
        "Educational project (OSU BDAA). Not a medical device. Do not upload identifiable "
        "patient data."
    )
    if bundle is None:
        with gr.Blocks(title="BoneScope") as demo:
            gr.Markdown(header)
            gr.Markdown(
                "⚠️ **No model loaded.** Set `BONESCOPE_MODEL` (local path) or "
                "`BONESCOPE_HF_REPO` (Hugging Face repo), or place `bonescope_model.pt` "
                "next to `app.py`."
            )
        return demo

    examples_dir = HERE / "examples"
    examples = sorted(str(p) for p in examples_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)

    with gr.Blocks(title="BoneScope") as demo:
        gr.Markdown(header)
        gr.Markdown(
            f"Model: `{bundle.source}` · {len(bundle.classes)} classes · "
            f"evaluation split: {bundle.split_caveat} · temperature {bundle.temperature:.2f}"
        )
        with gr.Tab("Single cell"):
            with gr.Row():
                inp = gr.Image(type="pil", label="Single-cell crop (Wright-Giemsa)")
                with gr.Column():
                    top3 = gr.Label(num_top_classes=3, label="Top-3 (calibrated probability)")
                    status = gr.Markdown()
            cam = gr.Image(label="Grad-CAM for the top prediction", type="numpy")
            gr.Button("Classify", variant="primary").click(
                lambda im: predict_single(bundle, im) if im is not None else ({}, None, ""),
                inp,
                [top3, cam, status],
            )
            if examples:
                gr.Examples(examples, inp)
        with gr.Tab("Batch (zip of crops)"):
            zf = gr.File(label="Zip of single-cell crops", file_types=[".zip"])
            table = gr.Dataframe(
                value=pd.DataFrame(columns=["class", "cells", "share"]), label="Cell counts"
            )
            summary = gr.Markdown()
            gr.Button("Analyse", variant="primary").click(
                lambda f: (
                    predict_batch(bundle, f.name if hasattr(f, "name") else f)
                    if f is not None
                    else (pd.DataFrame(), "")
                ),
                zf,
                [table, summary],
            )
        gr.Markdown(f"---\n*{DISCLAIMER}*")
    return demo


def main() -> None:
    path = resolve_model_path()
    bundle = load_bundle(path) if path else None
    build_demo(bundle).launch()


if __name__ == "__main__":
    main()
