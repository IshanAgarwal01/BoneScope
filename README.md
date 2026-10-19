# BoneScope: MDS bone marrow cell classifier

> **Educational project. Not a medical device. Not for clinical use.**

BoneScope classifies single Wright-Giemsa-stained bone marrow cell crops from
patients with myelodysplastic syndrome (MDS), evaluates the model with
leakage-aware splits, and shows how per-cell predictions could become a blast
percentage and a WHO 2022 blast category on synthetic "virtual patients".

Status: **work in progress.** Built phase by phase for the OSU BDAA ML track. Results
will appear here only after they have actually been run.

## Dataset

Shen D. et al., "A large dataset of bone marrow cells in myelodysplastic syndrome
for classification systems," *Scientific Data* 12, 1849 (2025).
https://www.nature.com/articles/s41597-025-06027-z ·
data: https://doi.org/10.6084/m9.figshare.28737170

The data is not included in this repo. See [`data/README.md`](data/README.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
pytest
```

## License

Code: MIT (see `LICENSE`). The dataset has its own license from its authors.
