"""Phase 1 data report: structure, classes, image stats, metadata, and ID evidence.

Usage:
    python scripts/inspect_data.py --root data/raw

Writes:
    reports/data_report.md          human-readable report (also printed)
    reports/class_counts.csv        images per class
    reports/figures/class_grid.png  8 random examples per class
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from mds.data import (  # noqa: E402
    filename_skeleton,
    index_images,
    keyword_hits,
    token_position_stats,
)

METADATA_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".txt", ".md"}
RARE_THRESHOLD = 20


def folder_tree(root: Path, max_depth: int = 3, max_children: int = 40) -> list[str]:
    """Directory tree with file counts, truncated so huge folders stay readable."""
    lines = []

    def walk(d: Path, depth: int) -> None:
        children = sorted(d.iterdir())
        dirs = [c for c in children if c.is_dir()]
        n_files = sum(1 for c in children if c.is_file())
        rel = d.relative_to(root).as_posix() if d != root else "."
        lines.append(f"{'  ' * depth}{rel}/  ({n_files} files, {len(dirs)} subfolders)")
        if depth + 1 > max_depth:
            return
        for sub in dirs[:max_children]:
            walk(sub, depth + 1)
        if len(dirs) > max_children:
            lines.append(f"{'  ' * (depth + 1)}... {len(dirs) - max_children} more folders")

    walk(root, 0)
    return lines


def describe_metadata(root: Path) -> list[str]:
    """List non-image files; for tables, show their columns and a few rows."""
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in METADATA_EXTS:
            continue
        rel = path.relative_to(root).as_posix()
        out.append(f"### `{rel}` ({path.stat().st_size:,} bytes)")
        try:
            if path.suffix.lower() in {".csv", ".tsv"}:
                df = pd.read_csv(path, sep=None, engine="python", nrows=2000)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path, nrows=2000)
            else:
                text = path.read_text(errors="replace")
                out.append("```\n" + text[:1500] + "\n```")
                continue
            out.append(f"Columns: {list(df.columns)}")
            out.append("```\n" + df.head(5).to_string() + "\n```")
            hits = keyword_hits(map(str, df.columns))
            if hits:
                out.append(f"**ID-like column names:** {dict(hits)}")
        except Exception as exc:  # report, don't crash: unknown formats are evidence too
            out.append(f"(could not parse: {exc})")
    return out or ["No metadata files found (only images)."]


def jpeg_fingerprint(img: Image.Image) -> str:
    """Hash of the JPEG quantisation tables.

    Different cameras/software/quality settings write different tables, so a
    handful of distinct fingerprints can hint at acquisition batches.
    """
    q = getattr(img, "quantization", None)
    if not q:
        return "n/a"
    blob = repr(sorted((k, tuple(v)) for k, v in q.items())).encode()
    return hashlib.md5(blob).hexdigest()[:8]


def image_stats(root: Path, paths: list[str]) -> pd.DataFrame:
    """Read only headers (fast): size, mode, format, JPEG fingerprint."""
    rows = []
    for rel in paths:
        with Image.open(root / rel) as img:
            rows.append(
                {
                    "path": rel,
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                    "jpeg_fp": jpeg_fingerprint(img),
                }
            )
    return pd.DataFrame(rows)


def class_grid(root: Path, df: pd.DataFrame, out: Path, per_class: int = 8, seed: int = 0) -> None:
    """Grid of `per_class` random images per class, classes sorted by size."""
    rng = random.Random(seed)
    classes = df["label_name"].value_counts().index.tolist()
    fig, axes = plt.subplots(
        len(classes), per_class, figsize=(per_class * 1.3, len(classes) * 1.45), squeeze=False
    )
    for r, cls in enumerate(classes):
        paths = df.loc[df["label_name"] == cls, "path"].tolist()
        picks = rng.sample(paths, min(per_class, len(paths)))
        for c in range(per_class):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(picks):
                with Image.open(root / picks[c]) as img:
                    ax.imshow(img.convert("RGB"))
        axes[r][0].set_title(f"{cls} (n={len(paths)})", loc="left", fontsize=8, pad=2)
    fig.suptitle("Random examples per class (educational project, not for clinical use)", y=1.0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_report(root: Path, reports: Path, grid: bool = True, seed: int = 0) -> str:
    df = index_images(root)
    lines = ["# Phase 1 data report", "", f"Root: `{root}`", ""]

    lines += ["## Folder structure", "```", *folder_tree(root), "```", ""]

    counts = df["label_name"].value_counts()
    lines += [
        "## Classes",
        f"**{len(counts)} classes, {len(df):,} images** (read from the folder names).",
        "",
        "| # | class | images | share |",
        "|---|---|---:|---:|",
    ]
    for i, (name, n) in enumerate(counts.items(), 1):
        lines.append(f"| {i} | {name} | {n:,} | {n / len(df):.1%} |")
    rare = counts[counts < RARE_THRESHOLD]
    lines += [
        "",
        f"Classes with fewer than {RARE_THRESHOLD} images: "
        + (", ".join(f"{k} ({v})" for k, v in rare.items()) if len(rare) else "none"),
        "",
    ]
    reports.mkdir(parents=True, exist_ok=True)
    counts.rename_axis("label_name").rename("n_images").to_csv(reports / "class_counts.csv")

    stats = image_stats(root, df["path"].tolist())
    sizes = stats["width"].astype(str) + "x" + stats["height"].astype(str)
    lines += [
        "## Image formats and sizes",
        f"Extensions: {dict(Counter(Path(p).suffix.lower() for p in df['path']))}",
        f"PIL formats: {stats['format'].value_counts().to_dict()}",
        f"Colour modes: {stats['mode'].value_counts().to_dict()}",
        f"Square crops: {(stats['width'] == stats['height']).mean():.1%}",
        "",
        "Side length (px) summary:",
        "```",
        stats[["width", "height"]].describe().round(1).to_string(),
        "```",
        f"Most common sizes: {sizes.value_counts().head(10).to_dict()}",
        f"Distinct JPEG quantisation fingerprints: {stats['jpeg_fp'].nunique()} "
        f"-> {stats['jpeg_fp'].value_counts().head(10).to_dict()}",
        "",
    ]

    lines += ["## Metadata files", *describe_metadata(root), ""]

    stems = [Path(p).stem for p in df["path"]]
    rng = random.Random(seed)
    sample = rng.sample(df["path"].tolist(), min(30, len(df)))
    skeletons = Counter(filename_skeleton(s) for s in stems)
    lines += [
        "## Identifier evidence",
        "30 random example paths:",
        "```",
        *sample,
        "```",
        "Filename skeletons (letters->A, digit runs->D), most common:",
        "```",
        *(f"{k!r}: {v:,}" for k, v in skeletons.most_common(10)),
        "```",
        f"ID-like keywords in filenames: {dict(keyword_hits(stems)) or 'none'}",
        "",
        "Unique values per filename token position (a position with ~125 unique "
        "values would be a patient-ID candidate; ~1 per file is a running index):",
        "",
        "| position | unique values | numeric | most common examples |",
        "|---:|---:|---|---|",
    ]
    for t in token_position_stats(stems):
        lines.append(
            f"| {t.position} | {t.n_unique:,} | {t.all_numeric} | {', '.join(t.examples[:6])} |"
        )
    lines.append("")

    if grid:
        class_grid(root, df, reports / "figures" / "class_grid.png", seed=seed)
        lines.append("Class grid saved to `reports/figures/class_grid.png`.")

    text = "\n".join(lines) + "\n"
    (reports / "data_report.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/raw", type=Path)
    parser.add_argument("--reports", default="reports", type=Path)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    if not args.root.exists():
        raise SystemExit(f"{args.root} not found. See data/README.md for download steps.")
    print(build_report(args.root, args.reports, grid=not args.no_grid, seed=args.seed))


if __name__ == "__main__":
    main()
