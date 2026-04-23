# Detecting Structural Distress at Scale

**Course:** MUSA 6500 — Geospatial Machine Learning in Remote Sensing
**Authors:** Jason Fan, Henry Sywulak-Herr
**Date:** April 2026

---

## 1. Problem

Predict, per Philadelphia parcel, whether a structure is **distressed** (L&I-classified hazardous or imminently dangerous) or **stable**, from 0.3 m RGB aerial imagery alone. A building-permit signal is used only post-hoc to suppress false positives during full-city inference.

## 2. Data

- **Imagery.** Philadelphia 0.3 m 4-band aerial (RGB used), hosted on S3 as a COG, read lazily via xarray + rioxarray.
- **Labels.** Philadelphia L&I property-violation records, joined to parcel polygons.
  - `label = 0` stable, `label = 1` distressed, `label = 2` imminently dangerous (collapsed to 1 at training time).
  - `label_permit_flagged` — boolean for an active building permit, used only as a post-inference mask.
- **Prior.** Distress is rare in the wild — roughly **1.4%** of all parcels.
- **Chip geometry.** 130 ft (~40 m) fixed window around each parcel centroid.

## 3. Shared evaluation protocol

Identical splits across every approach, to keep comparisons honest:

| Split | Source | Purpose |
|------|--------|---------|
| Natural-distribution holdout | 10% random sample, real ~1.4% prior | Fair generalization test |
| Training pool | Remaining 90%, downsampled to 2:1 stable:distressed | Balanced classifier training |
| Val | 20% of training pool, stratified | Threshold tuning / early stop |

Metrics on the **natural-prior holdout** are the ground truth. Balanced-val F1 is misleading because the prior shift is ~70×.

---

## 4. Approaches

### 4.1 Approach A — Clay v1.5 fine-tune with MLP head (`main_v2.ipynb`)

| | |
|---|---|
| Backbone | Clay v1.5 MAE encoder (312 M params, ViT-L), **frozen** |
| Head | Linear 1024 → 512 → 256 → 1, BCE loss |
| Inputs | 130 ft chip, resized to 224×224, LINZ band normalization |
| Conditioning | LINZ platform (Clay's closest pretraining profile to 0.3 m aerial) |
| Optim | AdamW lr=5e-5, OneCycleLR warmup, bf16 AMP |
| Augmentation | H/V flip + 180° rotation + resized-crop + color-jitter |

**Stated rationale.** Exploit a foundation model pretrained on remote-sensing imagery; follow the "frozen backbone + small head" pattern that works for most downstream RS classification tasks.

**Outcome.** Training loss plateaued at ~0.636 (the class-prior entropy) within three epochs. A direct diagnostic — measuring `std_across_batch` of the 1024-dim CLS output — came back at **~0.03**, meaning Clay produced *near-constant* embeddings across visually different inputs. The encoder was not separating Philadelphia rooftops at all. Natural-holdout recall oscillated between 0.001 and 0.333 as a function of where the threshold happened to fall on pure noise.

Root cause: Clay v1.5 was pretrained on 0.5 m (LINZ) down through 10 m Sentinel-2. Philadelphia's 0.3 m aerial falls outside every platform in Clay's metadata, and the features don't transfer.

---

### 4.2 Approach B — ConvNeXt-Tiny with LLRD (`main_v3.ipynb`)

| | |
|---|---|
| Backbone | ConvNeXt-Tiny (28 M params), ImageNet-22k → 1k, **fully unfrozen** |
| Head | LayerNorm → Linear(768→256) → Linear(256→1) |
| Inputs | Same 130 ft chip at 224×224, ImageNet normalization |
| Optim | LLRD (decay=0.5/stage, head 5e-4 → stem 6.25e-5), OneCycleLR 10% warmup, bf16 AMP |
| Loss | BCE, patience=4 early stop |
| Augmentation | Same stack as v2 |

**Stated rationale.** Pivot away from satellite-foundation models. At 0.3 m, building rooftops look more like natural-image textures (shingles, asphalt, metal) than the satellite scenes Clay was pretrained on. LLRD keeps ImageNet's low-level filters stable while letting task-specific layers adapt.

**Outcome.** First version where training loss moved past the prior entropy and validation AUC became non-trivial. PR-curve threshold tuning produced an operating point that **transferred** to the natural-prior holdout. Framed in the notebook as the intended publishable result: for sub-meter urban RGB, ImageNet features beat satellite-foundation-model features.

---

### 4.3 Approach C — Clay v1.5 embeddings + classifier on top (`main_v4.ipynb`, current)

Clay's official [finetune-on-embeddings tutorial](https://clay-foundation.github.io/model/finetune/finetune-on-embeddings.html) does not do neural fine-tuning at all. It runs Clay as a pure feature extractor, caches CLS tokens, and trains `sklearn.ensemble.RandomForestClassifier` on them. v4 follows that recipe exactly, to answer: "Did v2 fail because we deviated from Clay's documented usage?"

| | |
|---|---|
| Backbone | Clay v1.5 large, loaded via `ClayMAEModule.load_from_checkpoint` with prescribed `dolls=[16,32,64,128,256,768,1024]` |
| Feature vector (initial) | 1024-dim CLS token per chip |
| Feature vector (improved) | **3072-dim** `[CLS \| mean(patch) \| max(patch)]` with **5× rotation+flip TTA averaging** |
| Classifier | RF(n_est=500, class_weight="balanced"); compared against `LogisticRegression` + `CalibratedClassifierCV(RF, isotonic)` |
| Conditioning | LINZ band stats, RGB waves = [0.635, 0.555, 0.465], GSD = 0.3 m (native) |

**Stated rationale.** Do exactly what Clay's authors prescribe. No LLRD, no neural heads, no gradient-based training of the encoder.

---

## 5. v4 diagnostic results

### Validation confusion matrix (threshold 0.5)

![Validation confusion matrix — all predictions collapse to Stable](images/validation_cm.png)

At RF's default 0.5 cutoff the classifier predicts **zero** distressed chips. All 1,437 positives in the balanced val set are misclassified. This is not a threshold-tuning problem — it is the classifier telling us the features cannot separate the classes.

### Highest-confidence "distressed" predictions

![Top true-positive chip grid — highest predicted probabilities ~0.47](images/tp_grid.png)

The most-confident distressed scores top out at **p ≈ 0.44–0.47**, all below the 0.5 decision boundary. Visually, these are ordinary row-house rooftops, indistinguishable from neighbors labeled stable.

### t-SNE projection of CLS embeddings (validation)

![t-SNE — stable and distressed fully interleaved](images/tsne.png)

Stable (blue) and distressed (red) are fully interleaved. There is no region of Clay's embedding space that concentrates distressed parcels. No classifier on these features can separate the classes — a Random Forest, a linear probe, a neural head, or anything else would hit the same wall.

### Nearest neighbors of a distressed query chip

![Nearest neighbors — 5 of 6 neighbors are stable despite tiny cosine distance](images/nearest_neighbors.png)

The six chips closest in cosine distance to a distressed query include only **one** other distressed example and five stable ones. The visual similarity is real — all pale flat roofs — which is exactly the point: Clay's features capture roof material and geometry faithfully, but not the condition signal needed for this task.

### Natural-distribution confusion matrix (tuned threshold)

![Natural-prior CM — flips to predicting distressed on every single parcel](images/natural_cm.png)

When the threshold is pushed low enough to produce any distressed predictions at all, it flips to predicting distress on **every** parcel (54,728 positive vs 0 negative). There is no threshold between these two collapses that yields a usable classifier — consistent with the t-SNE showing no separating signal exists.

---

## 6. Conclusion

Across three architectures, the finding is consistent: **Clay v1.5's features do not encode the fine-grained roof-condition signals needed to distinguish distressed from stable Philadelphia parcels at 0.3 m resolution.**

- **v2 (Clay fine-tune, frozen encoder + MLP head):** classifier collapsed; `std_across_batch ≈ 0.03` showed the features were near-constant.
- **v3 (ConvNeXt-Tiny + LLRD):** the only approach that produced a coherent natural-prior F1. ImageNet pretraining transfers better than Clay for sub-meter urban RGB.
- **v4 (Clay embeddings + RF, Clay's canonical recipe):** even with a 3072-dim multi-token readout and 5× TTA, the classes are not separable (t-SNE), and the classifier collapses either to all-stable (τ=0.5) or all-distressed (tuned τ).

**Primary claim.** For 0.3 m single-date aerial RGB, a satellite-foundation-model's strength (scale-aware ViT trained across sensors, 0.5 m–10 m GSD) is the wrong prior. Clay's pretraining distribution never sees individual shingles, gutters, fascia, or roof decay at this scale. **Pretraining distance, not classifier capacity, is the bottleneck.** No amount of downstream head engineering bridges it — three different heads (MLP, linear probe, calibrated RF) all fail the same way on Clay features, while a general-purpose ImageNet backbone produces a working classifier on the same pixels.

**Secondary claim.** The distress label itself is partially unobservable from directly overhead. L&I violations include foundation cracks, interior water damage, and failed egress — signals invisible in a top-down RGB chip. The nearest-neighbor analysis bounds how much any aerial-RGB classifier can hope to achieve: Clay reliably retrieves visually similar parcels, but those parcels carry mixed labels because the label is not fully determined by what the imagery shows.

**Recommended path forward.**

1. Ship **v3 (ConvNeXt-Tiny)** as the working classifier. It is the only one of the three that produces a usable operating point on the natural prior.
2. For more signal, fuse in modalities the label actually depends on: oblique imagery (Pictometry) for facade-visible damage, multi-date differencing to catch new roof tarps / blue tarps / debris accumulation, or non-imagery signals (age, last-inspection date, permit history).
3. Treat "Clay doesn't work here" as a **publishable negative finding** about the transfer limits of Earth-observation foundation models into sub-meter urban condition-assessment tasks — not as a debugging failure.
