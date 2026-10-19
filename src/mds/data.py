"""Image indexing, filename analysis, and proxy grouping.

Phase 1 only needs indexing and grouping; the Dataset, transforms and samplers
are added in Phase 2.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Words that would suggest a patient/slide/case identifier in a filename.
ID_KEYWORDS = re.compile(
    r"(patient|pat|pt|case|slide|sample|subject|pid|sid|mrn|smear|field|fov)",
    re.IGNORECASE,
)
_TOKEN_SPLIT = re.compile(r"[_\-\s.()]+")


def index_images(root: str | Path) -> pd.DataFrame:
    """List every image under `root` with its class taken from the parent folder.

    Assumes the common "one folder per class" layout. Paths are stored relative
    to `root` so splits.csv stays portable between laptop, cloud and Colab.
    """
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "label_name": path.parent.name,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No images with extensions {sorted(IMAGE_EXTS)} under {root}")
    return pd.DataFrame(rows)


def filename_skeleton(stem: str) -> str:
    """Collapse a filename into a pattern: letters -> 'A', digit runs -> 'D'.

    e.g. "P012_S3_cell0045" -> "AD_AD_AD". Many files sharing one skeleton
    with a variable digit field is how embedded IDs usually show up.
    """
    s = re.sub(r"[A-Za-z]+", "A", stem)
    return re.sub(r"\d+", "D", s)


@dataclass
class TokenStats:
    position: int
    n_unique: int
    examples: list[str]
    all_numeric: bool


def token_position_stats(stems: Iterable[str], max_examples: int = 8) -> list[TokenStats]:
    """Count unique values at each token position of the filenames.

    A position with roughly as many unique values as there are patients
    (~125 here) is a candidate patient ID; a position unique per file is just a
    running index.
    """
    split = [_TOKEN_SPLIT.split(s) for s in stems]
    width = max((len(t) for t in split), default=0)
    stats = []
    for i in range(width):
        values = [t[i] for t in split if len(t) > i and t[i]]
        counts = Counter(values)
        stats.append(
            TokenStats(
                position=i,
                n_unique=len(counts),
                examples=[v for v, _ in counts.most_common(max_examples)],
                all_numeric=bool(values) and all(v.isdigit() for v in values),
            )
        )
    return stats


def keyword_hits(names: Iterable[str]) -> Counter:
    """Count ID-like keywords (patient, slide, case, ...) in filenames or columns."""
    hits: Counter = Counter()
    for name in names:
        for match in ID_KEYWORDS.findall(name):
            hits[match.lower()] += 1
    return hits


def border_color_stats(img: Image.Image, border: int = 4) -> np.ndarray:
    """Median and IQR of the RGB values in a thin frame around the crop.

    Crops are the cell box expanded by 10%, so the frame is mostly background
    (plasma/red cells/glass). Its colour is set by the slide's stain batch and
    the microscope's lighting rather than by the cell type, which makes it a
    reasonable fingerprint for "came from the same slide/session".
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    b = max(1, min(border, arr.shape[0] // 4, arr.shape[1] // 4))
    frame = np.concatenate(
        [
            arr[:b].reshape(-1, 3),
            arr[-b:].reshape(-1, 3),
            arr[:, :b].reshape(-1, 3),
            arr[:, -b:].reshape(-1, 3),
        ]
    )
    q25, q50, q75 = np.percentile(frame, [25, 50, 75], axis=0)
    return np.concatenate([q50, q75 - q25]).astype(np.float32)


def build_proxy_features(
    embeddings: np.ndarray,
    color_stats: np.ndarray,
    n_pca: int = 32,
    color_weight: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Combine appearance embeddings with background colour into one feature.

    Both blocks are standardised so neither dominates by scale; colour is then
    up-weighted because it tracks the slide, whereas the CNN embedding also
    tracks the cell type (which we do NOT want clusters to follow).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_pca = min(n_pca, embeddings.shape[0], embeddings.shape[1])
    emb = PCA(n_components=n_pca, random_state=seed).fit_transform(embeddings)
    emb = StandardScaler().fit_transform(emb) / np.sqrt(n_pca)
    col = StandardScaler().fit_transform(color_stats) / np.sqrt(color_stats.shape[1])
    return np.hstack([emb, color_weight * col]).astype(np.float32)


def cluster_proxy_groups(features: np.ndarray, n_groups: int = 125, seed: int = 42) -> np.ndarray:
    """K-means the proxy features into `n_groups` pseudo-patients."""
    from sklearn.cluster import KMeans

    n_groups = min(n_groups, len(features))
    km = KMeans(n_clusters=n_groups, n_init=10, random_state=seed)
    return km.fit_predict(features)


def cluster_class_purity(groups: np.ndarray, labels: np.ndarray) -> float:
    """Image-weighted share of each cluster's majority class.

    If this is close to 1, the clusters are just re-discovering cell types, not
    slides, and the proxy grouping would make stratified folds impossible.
    Compare it to the purity expected by chance (the largest class share).
    """
    df = pd.DataFrame({"g": groups, "y": labels})
    majority = df.groupby("g")["y"].agg(lambda s: s.value_counts().iloc[0])
    return float(majority.sum() / len(df))
