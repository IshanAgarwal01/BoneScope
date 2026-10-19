"""Image indexing, filename analysis, proxy grouping, transforms, Dataset, samplers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF

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


# --- transforms, dataset, samplers, stain normalisation ---------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomRot90:
    """Rotate by 0/90/180/270 degrees.

    A cell on a smear has no "up", so every right-angle rotation is an equally
    valid view. Right angles avoid the interpolation blur and black corners
    that arbitrary-angle rotation would add.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        k = int(torch.randint(0, 4, (1,)))
        return img.rotate(90 * k) if k else img


class FixedScalePad:
    """Place the crop on a fixed canvas WITHOUT rescaling cells to fill it.

    Plain resizing blows small cells up and shrinks large ones, so every cell
    ends up the same size on screen, yet cell size is diagnostic (e.g. a
    lymphocyte vs a blast vs a megakaryocyte). This mode scales every image by
    one global factor (img_size / canvas_px) and pads with the crop's own
    background colour, so relative size survives. Crops bigger than the canvas
    are shrunk to fit (rare when canvas_px is set near the 99th percentile).
    """

    def __init__(self, img_size: int, canvas_px: int) -> None:
        self.img_size = img_size
        self.factor = img_size / canvas_px

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.factor, self.img_size / max(w, h))
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = img.resize(new, Image.BILINEAR)
        fill = tuple(int(v) for v in border_color_stats(img)[:3])
        canvas = Image.new("RGB", (self.img_size, self.img_size), fill)
        canvas.paste(img, ((self.img_size - new[0]) // 2, (self.img_size - new[1]) // 2))
        return canvas


def _size_transform(img_size: int, resize_mode: str, canvas_px: int | None):
    if resize_mode == "resize":
        # Crops are already square (box + 10%), so this does not distort shape.
        return T.Resize((img_size, img_size), antialias=True)
    if resize_mode == "fixed_scale":
        if not canvas_px:
            raise ValueError("resize_mode=fixed_scale needs data.canvas_px")
        return FixedScalePad(img_size, canvas_px)
    raise ValueError(f"Unknown resize_mode {resize_mode!r}")


def build_transforms(
    img_size: int,
    train: bool,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    resize_mode: str = "resize",
    canvas_px: int | None = None,
) -> T.Compose:
    """Training augmentation that respects what a haematologist looks at.

    Included: flips and right-angle rotations (orientation is meaningless),
    a small shift (the detector's box is not perfectly centred), mild
    colour jitter (stain intensity varies between slides) and an occasional
    light blur (focus varies).

    Deliberately excluded:
      * strong hue shifts / grayscale: Wright-Giemsa colour carries meaning
        (basophilic blue cytoplasm in blasts, pink granules, purple nuclei).
      * random-resized-crop / scale jitter: cell and nucleus size are
        diagnostic, and cropping could cut away the nucleus or granules.
      * CutMix / MixUp: pasting half of one cell onto another creates
        morphology that does not exist and blurs the class boundary.
    """
    steps: list = [_size_transform(img_size, resize_mode, canvas_px)]
    if train:
        shift = max(1, int(0.05 * img_size))
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRot90(),
            # Reflect-pad then crop back = a small shift without black borders.
            T.Pad(shift, padding_mode="reflect"),
            T.RandomCrop(img_size),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.2),
        ]
    steps += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(steps)


class ReinhardStainNormalizer:
    """Reinhard colour normalisation (match LAB mean/std to a reference crop).

    Off by default. It can remove stain variation that confuses the model,
    but it can also wash out real colour differences between cell types, so
    look at before/after examples (scripts/stain_preview.py) before enabling.
    """

    def __init__(self, reference: Image.Image) -> None:
        import torchstain

        self._norm = torchstain.normalizers.ReinhardNormalizer(backend="numpy")
        self._norm.fit(np.asarray(reference.convert("RGB")))

    def __call__(self, img: Image.Image) -> Image.Image:
        out = self._norm.normalize(np.asarray(img.convert("RGB")))
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


class MarrowDataset(Dataset):
    """Single-cell crops listed in a splits DataFrame (columns: path, label)."""

    def __init__(
        self,
        df: pd.DataFrame,
        root: str | Path,
        transform=None,
        stain_normalizer: ReinhardStainNormalizer | None = None,
    ) -> None:
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.root = Path(root)
        self.transform = transform
        self.stain_normalizer = stain_normalizer

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, idx: int) -> Image.Image:
        with Image.open(self.root / self.paths[idx]) as img:
            img = img.convert("RGB")
        if self.stain_normalizer is not None:
            img = self.stain_normalizer(img)
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.load(idx)
        x = self.transform(img) if self.transform else TF.to_tensor(img)
        return x, self.labels[idx]


def sqrt_inverse_sampler(labels: list[int] | np.ndarray, seed: int = 42) -> WeightedRandomSampler:
    """Sample each image with probability proportional to 1/sqrt(class count).

    Full inverse-frequency sampling would show a 20-image class as often as a
    3,000-image class, i.e. the same 20 images over and over (overfitting).
    The square root is a compromise: rare classes are seen ~sqrt(ratio) times
    more often than before, not ratio times.
    """
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels)
    weights = 1.0 / np.sqrt(counts[labels])
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True, generator=gen
    )
