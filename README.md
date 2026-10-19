# BoneScope: MDS bone marrow cell classifier

> **Educational project. Not a medical device. Not for clinical use.**

**Problem, in one sentence:** can a deep network classify single Wright-Giemsa bone
marrow cells from patients with myelodysplastic syndrome (MDS), and be evaluated
honestly enough that its blast count could one day be trusted?

BoneScope trains a cell classifier, evaluates it with leakage-aware cross-validation
(macro-F1 first), calibrates and explains it, turns per-cell predictions into a blast
percentage and WHO 2022 blast range on synthetic "virtual patients", and serves it in a
Gradio demo. Built for the Ohio State University BDAA ML track.

## Status

| Phase | What | State |
|---|---|---|
| 0 | setup, configs, utilities | done |
| 1 | data download + inspection + grouping | **code done; waiting on data** (figshare is blocked in the build environment) |
| 2 | frozen splits, leakage checks, augmentation, sampler | code done + tested |
| 3 | models, losses, two-stage training, smoke test | code done + tested on synthetic images |
| 4 | experiments (5-fold CV) | **not run yet** (needs data + Colab GPU) |
| 5 | per-class metrics, calibration, Grad-CAM | code done + tested |
| 6 | blast %, bootstrap CI, WHO 2022, virtual patients | code done + tested |
| 7 | Gradio app | done; tested with a synthetic model |
| 8 | docs + slides | outline done; results sections pending |

**No result on real data has been produced yet.** Anything in this repo that looks like a
number about cells came from synthetic test images and means nothing.

## Dataset

Shen D. et al., "A large dataset of bone marrow cells in myelodysplastic syndrome for
classification systems," *Scientific Data* 12, 1849 (2025).
https://www.nature.com/articles/s41597-025-06027-z · data:
https://doi.org/10.6084/m9.figshare.28737170

- 25,009 single-cell crops from 125 MDS patients (WHO 2022), Wright-Giemsa stain;
  crops are the cell's box expanded by 10%.
- Experts annotated 32 categories; the release tags 27. The code reads the class list
  from the files and never hard-codes it.
- A 2026 follow-up (https://link.springer.com/article/10.1007/s44443-026-00948-4) reports
  no patient identifiers in the public release. `scripts/inspect_data.py` checks this
  directly from filenames and metadata.
- The data is **not** in this repo (see [`data/README.md`](data/README.md)); it is under
  its authors' license. `scripts/download_data.py` prints that license.

## Why the evaluation is built the way it is

- **Leakage.** Cells from the same patient share stain, smear quality and lighting. If one
  patient's cells sit in both train and test, the model can score well by recognising the
  slide rather than the cell. Folds use `StratifiedGroupKFold(5)` over patients if IDs
  exist, otherwise over **proxy groups** (k-means on CNN features + background colour,
  ~125 clusters). A plain image-level split is reported alongside for comparison. Group
  disjointness is asserted when splits are made, when they are loaded, at training start,
  and in tests.
- **Honest labels.** Proxy and image-level results are labelled *"may be optimistic"*
  everywhere they appear.
- **Macro-F1 first.** Classes are very imbalanced; macro-F1 weighs a rare dysplastic class
  as much as a common one. Checkpoints are selected on a separate validation split, never
  on the test fold. Accuracy is reported only for comparison with the published 80.13%
  (dataset paper) and 81.65% / 0.716 macro-F1 (2026 paper, image-level CV, 23 classes).
  A macro-F1 far above ~0.75 would make us look for leakage or a bug first.
- **Augmentation respects morphology.** Flips, right-angle rotations, small shifts and mild
  colour jitter only. No hue shifts or grayscale (Giemsa colour is diagnostic), no
  random-resized crops (cell size is diagnostic), and no CutMix (it creates impossible
  cells). An optional fixed-scale mode keeps relative cell size.

## Reproduce

```bash
git clone https://github.com/IshanAgarwal01/BoneScope.git && cd BoneScope
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . --no-deps
pytest                                               # includes a CPU training smoke test

python scripts/download_data.py --out data           # or download manually -> data/raw/
python scripts/inspect_data.py --root data/raw       # Phase 1 report + class grid
python scripts/compute_proxy_groups.py --root data/raw          # only if no patient IDs
python scripts/make_splits.py --root data/raw --groups-csv splits/proxy_groups.csv --split-type proxy
bash scripts/run_cv.sh configs/convnext_tiny.yaml output.run_name=convnext_ce
python scripts/analyze_run.py --run runs/convnext_ce/proxy --blast-classes <names> --non-nucleated <names>
python scripts/gradcam_gallery.py --run runs/convnext_ce/proxy --fold 0
```

No local GPU? Use [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb), which does all
of the above on a free Colab T4 and saves runs to Google Drive.

Every hyperparameter is in [`configs/`](configs); override any one from the command line,
e.g. `python -m mds.train --config configs/convnext_tiny.yaml --fold 0 --set loss.name=focal`.

## Results

**Not run yet.** This table will be filled from `reports/results.md` (mean ± std over 5
folds) once Phase 4 has run, with the split type in every row.

| model | loss / sampler | split | macro-F1 | balanced acc | top-3 | accuracy |
|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | CE + LS 0.1 | proxy (may be optimistic) | - | - | - | - |
| ConvNeXt-Tiny | CE + LS 0.1 | image-level (may be optimistic) | - | - | - | - |

Figures (generated by the scripts; pending): class grid, confusion matrix, reliability
diagram, Grad-CAM gallery and virtual-patient plot, all in `reports/figures/`.

## Clinical translation (demo only)

`src/mds/clinical.py` computes blast % = blasts / nucleated cells (hard counts and
probability-weighted soft counts), a 1,000-resample bootstrap 95% CI, and the WHO 2022
range with its ICC 2022 equivalent:

| marrow blasts | WHO 2022 | ICC 2022 |
|---|---|---|
| < 5% | MDS-LB | MDS, NOS / other <5% categories |
| 5-9% | MDS-IB1 | MDS-EB |
| 10-19% | MDS-IB2 | MDS/AML |
| >= 20% | AML | AML |

A case is flagged **borderline** when its CI crosses 5, 10 or 20%. It is demonstrated on
virtual patients: 500-cell mixtures of held-out test cells with known blast fractions
(2-25%). This dataset is a curated set of crops, not a consecutive 500-cell differential,
so a per-patient blast % computed from it would not be meaningful.

## Limitations

- **Split caveat:** without patient IDs, proxy groups only approximate patient-level
  separation; reported scores may be optimistic.
- **No iron stain:** ring sideroblasts need Prussian blue, so MDS-SF3B1 / ring-sideroblast
  assessment is impossible here and not attempted.
- **Single centre, single stain protocol:** expect a drop on other labs' images.
- **Curated crops, not a differential:** class frequencies do not reflect a real marrow.
- **Blast count is one axis:** WHO 2022 also needs blood counts, Auer rods, cytogenetics and
  mutations.
- **Not a medical device:** for teaching and research only.

## Future work

- Iron-stained data, so ring sideroblasts can be assessed.
- Whole-slide cell detection (YOLO / Faster R-CNN) feeding the classifier, giving true
  consecutive-cell differentials.
- External validation on another hospital's smears; a patient-ID mapping from the
  authors (a draft request is in [`reports/email_to_authors.md`](reports/email_to_authors.md)).

## Repository map

```
configs/        YAML configs (base + ConvNeXt-Tiny, EfficientNet-B3, DINOv2 probe)
src/mds/        config, data, splits, models, losses, train, evaluate,
                calibration, explain (Grad-CAM), clinical, utils
scripts/        download, inspect, proxy groups, splits, stain preview, CV, aggregation,
                analysis, Grad-CAM gallery, model export
tests/          91 tests incl. leakage checks and CPU training smoke tests
notebooks/      EDA, results, Grad-CAM gallery, Colab GPU runner
app/            Gradio demo + Hugging Face Space card
reports/        results, figures, slide outline, email draft
```

## License

Code: MIT (see `LICENSE`). The dataset is the authors' and is under its own license.
