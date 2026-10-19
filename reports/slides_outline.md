# BoneScope: slide outline (~10 min, BDAA)

> Slots marked **[RESULT]** must be filled from `reports/results.md` and
> `reports/analysis_*.md` after the experiments run. Do not present placeholder numbers.

**1. Title.** BoneScope: classifying bone marrow cells in MDS, honestly.
Team names · OSU BDAA ML track · "Educational project, not a medical device."

**2. The clinical problem (1 min).** MDS is a group of bone-marrow cancers where blood cells
grow abnormally. Diagnosis and risk depend on the **blast percentage** from counting
500+ marrow cells by eye: slow and variable between observers. WHO 2022 cut-offs are
5%, 10% and 20%; one cell type can change the category. *Visual:* a smear and the
threshold bar (MDS-LB / IB1 / IB2 / AML, ICC in brackets).

**3. The data.** Shen et al., *Sci Data* 2025: 25,009 crops, 125 patients, Wright-Giemsa,
[N classes read from the files]. *Visual:* `class_grid.png` + a log-scale class bar
chart. Message: extreme imbalance → macro-F1.

**4. Finding 1: the leakage trap (lead with this).** The public release has no patient
IDs [confirm from the inspector's output]. Random image splits put the same patient in
train and test, so the model can memorise slides. Our fix: proxy groups (k-means on
image features + background colour) → `StratifiedGroupKFold`; we report both splits and
label them. *Visual:* a diagram of patient A's cells on both sides of the split.

**5. Finding 2: what the stain can't show.** Ring sideroblasts need an iron stain; this
dataset is Wright-Giemsa only → no MDS-SF3B1/RS claims. Colour carries diagnosis → no
hue/grayscale augmentation; size is diagnostic → no random-resized crops.

**6. Method.** ConvNeXt-Tiny (ImageNet-22k) → linear probe → full fine-tune with
discriminative LRs; selection on validation macro-F1; comparisons: EfficientNet-B3,
frozen DINOv2. Imbalance experiments: CE vs class-balanced CE vs focal vs sqrt sampler,
one change at a time. *Visual:* pipeline diagram.

**7. Results. [RESULT]** Table: macro-F1, balanced accuracy, top-3, accuracy (mean ± std,
5 folds) for proxy vs image-level splits, next to the published 80.13% / 81.65%
accuracy and 0.716 macro-F1. Say out loud which split each row used.

**8. Where it fails. [RESULT]** Row-normalised confusion matrix, top confusions
(expected: maturation neighbours; dysplastic vs normal granulocytes, which the authors
report at F1 ≈ 0.31). Blast precision/recall called out on its own.

**9. Can we trust the confidence? [RESULT]** Reliability diagram before/after temperature
scaling, ECE numbers. Grad-CAM: good examples plus any background "shortcut" cases.

**10. From cells to a category: virtual patients. [RESULT]** True vs predicted blast % for
2-25% mixtures with bootstrap CIs and WHO lines; borderline cases flagged. Emphasise:
synthetic mixtures, because this dataset is not a 500-cell differential.

**11. Demo.** Gradio app: one crop → top-3 + Grad-CAM + low-confidence banner; a zip →
counts, blast % with CI, WHO 2022 (ICC) range, borderline flag. Disclaimer on screen.

**12. Limitations and next steps.** Proxy split may be optimistic; single centre; no iron
stain; curated crops; not a medical device. Next: patient IDs from the authors,
external validation, whole-slide detection (YOLO / Faster R-CNN), iron-stained data.
