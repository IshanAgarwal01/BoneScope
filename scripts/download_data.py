"""Download the MDS bone marrow dataset from figshare and verify checksums.

Uses the public figshare API (no login). Works on Colab or any machine whose
network allows figshare; it will NOT work behind a proxy that blocks it.

Usage:
    python scripts/download_data.py --out data            # list + download + extract
    python scripts/download_data.py --out data --list     # only list the files

Files land in <out>/downloads/; archives are extracted to <out>/raw/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ARTICLE_ID = 28737170  # doi:10.6084/m9.figshare.28737170
API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
HEADERS = {"User-Agent": "BoneScope-edu/0.1 (research download)"}


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return json.load(r)


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        done = 0
        while block := r.read(1 << 20):
            f.write(block)
            done += len(block)
            if size:
                print(f"\r  {dest.name}: {done / 1e6:,.0f} / {size / 1e6:,.0f} MB", end="")
    print()
    tmp.rename(dest)


def safe_extract(archive: Path, out: Path) -> None:
    """Extract zip/tar, refusing members that would escape `out` (zip-slip)."""
    out = out.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():
                if not (out / m).resolve().is_relative_to(out):
                    raise RuntimeError(f"unsafe path in archive: {m}")
            z.extractall(out)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")
    else:
        shutil.copy2(archive, out / archive.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data", type=Path)
    parser.add_argument("--list", action="store_true", help="only list files")
    args = parser.parse_args()

    meta = fetch_json(API)
    print(
        f"{meta.get('title')}\nlicense: {meta.get('license', {}).get('name')}"
        f" ({meta.get('license', {}).get('url')})"
    )
    files = meta.get("files", [])
    for f in files:
        print(f"  {f['name']}  {f['size'] / 1e6:,.1f} MB  md5={f.get('computed_md5')}")
    if args.list:
        return

    downloads = args.out / "downloads"
    raw = args.out / "raw"
    downloads.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    (args.out / "figshare_metadata.json").write_text(json.dumps(meta, indent=2))
    for f in files:
        dest = downloads / f["name"]
        expected = f.get("computed_md5")
        if dest.exists() and expected and md5sum(dest) == expected:
            print(f"  {dest.name}: already downloaded, checksum OK")
        else:
            download(f["download_url"], dest, f["size"])
            if expected and md5sum(dest) != expected:
                raise SystemExit(f"Checksum mismatch for {dest.name}; delete it and retry")
        safe_extract(dest, raw)
    print(f"done -> {raw}. Next: python scripts/inspect_data.py --root {raw}")


if __name__ == "__main__":
    main()
