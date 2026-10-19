---
title: BoneScope
emoji: 🔬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Educational MDS bone marrow cell classifier (not clinical)
---

# BoneScope: MDS bone marrow cell classifier

> **Research demo. Not for clinical use. Trained on Wright-Giemsa single-cell crops.**
> Educational project (Ohio State University, BDAA ML track). Not a medical device.

**Single cell:** upload a single-cell crop and see the top-3 cell types with
temperature-calibrated probabilities, a Grad-CAM heat map, and a low-confidence warning
when the top probability is under 50%.

**Batch:** upload a zip of crops and see the cell counts, a blast percentage
(blasts / nucleated cells) with a bootstrap 95% CI, the WHO 2022 blast range with its ICC 2022
equivalent, and a *borderline* flag when the CI crosses 5, 10 or 20%.

## Limitations (read these)

- The evaluation split is stated in the app header. If it says "proxy" or "image-level",
  reported accuracy **may be optimistic**: the public dataset has no patient IDs.
- No iron stain, so **no ring sideroblast / MDS-SF3B1 assessment**.
- Single-centre data from one stain protocol; expect worse results on other labs' images.
- The batch blast % only demonstrates the arithmetic. It is not a 500-cell differential,
  and WHO 2022 classification also needs blood counts, cytogenetics and mutations.

## Data and model

Trained on: Shen D. et al., "A large dataset of bone marrow cells in myelodysplastic
syndrome for classification systems," *Scientific Data* 12, 1849 (2025),
doi:10.6084/m9.figshare.28737170. The dataset is under its authors' license.

Code: https://github.com/IshanAgarwal01/BoneScope (MIT).

## Configuration

The app loads, in order: `BONESCOPE_MODEL` (path), then `BONESCOPE_HF_REPO` +
`BONESCOPE_HF_FILE` (a Hugging Face model repo), then `./bonescope_model.pt`.

## Deployment steps (do these yourself; nothing here deploys automatically)

1. Train and export a model (Colab notebook, section 6), then:
   `python scripts/export_model.py --run <runs/...> --fold <k> --blast-classes ... --non-nucleated ... --out app/bonescope_model.pt`
2. Create a free account at https://huggingface.co, then **New Space**: SDK *Gradio*,
   hardware *CPU basic*, visibility of your choice.
3. Upload weights (pick one):
   - small model (<1 GB is fine): add `bonescope_model.pt` to the Space itself (step 4), or
   - **New model** repo (e.g. `<you>/bonescope-weights`), upload `bonescope_model.pt`, then in
     the Space settings add the variable `BONESCOPE_HF_REPO=<you>/bonescope-weights`
     (and a secret `HF_TOKEN` if that repo is private).
4. Put these files at the **root** of the Space repo:
   ```
   app.py  requirements.txt  README.md   (from app/)
   mds/                                  (copy of src/mds/)
   examples/                             (only if the dataset license allows)
   bonescope_model.pt                    (unless loaded from a model repo)
   ```
   From a terminal:
   ```bash
   git clone https://huggingface.co/spaces/<you>/BoneScope hf-space
   cp app/app.py app/requirements.txt app/README.md hf-space/
   cp -r src/mds hf-space/mds
   cd hf-space && git lfs install && git lfs track "*.pt"
   git add . && git commit -m "Deploy BoneScope demo" && git push
   ```
5. Wait for the build log to say *Running*, then try one crop and one small zip.
