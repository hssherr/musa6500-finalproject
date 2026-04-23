# Three Models, One Lesson: What Happens When You Point a Geospatial Foundation Model at Philadelphia's Rooftops

*Jason Fan & Henry Sywulak-Herr*
*MUSA 6500 — Geospatial Machine Learning in Remote Sensing, University of Pennsylvania*

---

## The setup

Philadelphia Licenses and Inspections maintains a registry of buildings that are structurally compromised. Roughly 140 parcels at any given time carry the most severe designation, Imminently Dangerous, meaning the structure is at imminent risk of collapse. Another 7,800 are flagged Unsafe. Together they account for less than 1.5 percent of the city's 547,000 parcels, and the map looks like this: a handful of red and orange points scattered across a dense stable background.

![Labeled parcels across Philadelphia, with Imminently Dangerous (red) and Unsafe (orange) parcels visible as rare points against the stable background.](ID_property.png)

The registry is reactive. It responds to complaints, inspection triggers, and permit activity rather than any systematic visual review of the built environment. Meanwhile, the city publishes annual 0.25 ft (7.62 cm) aerial imagery of every parcel, refreshed with each flight. The obvious question is whether the gap can be closed computationally — whether a vision model can look at a rooftop and say "that one needs an inspector."

We spent a semester answering that question across three architectures. This post walks through all three. None of them produced a deployable classifier, but together they say something concrete about the transfer limits of Earth observation foundation models when pushed outside their pretraining distribution.

## Evaluation protocol, held constant across all three versions

Before any of the modeling, we fixed the data split so the three approaches would be directly comparable. Ten percent of all parcels were set aside as a natural-distribution holdout, preserving the ~1.4 percent distressed prior. The remaining 90 percent became the training pool, rebalanced to a 2:1 stable-to-distressed ratio and then split 80/20 stratified into train and validation folds. Every model sees exactly the same chips, the same labels, and the same geography. Balanced validation F1 is useful for threshold tuning, but the numbers that actually matter are always the ones on the natural-prior holdout, because that is the only split that reflects operational reality.

Chips are 130-foot windows around each parcel centroid, padded to square and resized to whatever input resolution the backbone expects. Labels collapse L&I's three-class coding into binary: zero for stable, one for anything the city has flagged as distressed.

## Version 2: Clay v1.5 with a frozen backbone and an MLP head

The first approach was the one the literature recommends for downstream tasks on a foundation model. We loaded Clay v1.5's MAE encoder — 312 million parameters, Vision Transformer-Large — froze it, and attached a small MLP head (1024 → 512 → 256 → 1) trained with BCE loss. Inputs were normalized with Clay's LINZ band statistics because LINZ, at 0.5 m, is the closest pretraining platform to Philadelphia's 0.3 m aerial. Optimization was AdamW at 5e-5 with a OneCycle warmup in bf16, and the augmentation stack included flips, 180-degree rotation, resized crops, and color jitter.

Training loss plateaued at approximately 0.636 within three epochs and refused to move. That number is suspicious on its own — it is essentially the entropy of the class prior — so we ran a diagnostic. For each batch, we measured `std_across_batch` on the 1024-dimensional CLS output: the dimension-wise standard deviation of the embeddings across visually different chips. The value came back at roughly 0.03. Clay was producing near-constant embeddings regardless of input. On the natural holdout, recall oscillated between 0.001 and 0.333 depending on where the threshold happened to fall on pure noise.

The root cause was not the head. Clay v1.5 was pretrained on imagery ranging from 0.5 m LINZ aerial down through 10 m Sentinel-2. Philadelphia's 0.3 m aerial is outside every platform in Clay's metadata. At that resolution, what fills a chip is no longer urban morphology in the satellite sense — it is individual shingles, gutter lines, and asphalt grain. The foundation model's filters have never been asked to care about those features, and they do not respond to them.

## Version 3: ConvNeXt-Tiny with layer-wise learning rate decay

The natural pivot from this failure was to abandon the satellite foundation model and try a general-purpose vision backbone. ConvNeXt-Tiny, 28 million parameters, pretrained on ImageNet-22k and fine-tuned to ImageNet-1k, is a standard choice for this kind of drop-in. We unfroze the full network, attached a lightweight head (LayerNorm → Linear(768→256) → Linear(256→1)), and used layer-wise learning rate decay with a decay factor of 0.5 per stage — head at 5e-4, stem at roughly 6.25e-5. Everything else matched v2: same chips, same augmentations, same loss, same splits.

This was the first version where training loss moved past the prior entropy. Validation AUC became non-trivial, and the PR-curve-tuned operating point transferred to the natural-prior holdout rather than collapsing on contact with the real class distribution. The result was not extraordinary — the model is still working against a label column that is only partially visible from overhead — but it was coherent.

The methodological point is worth stating plainly: for sub-meter urban RGB, ImageNet pretraining outperformed the satellite-foundation-model pretraining on the same pixels. At 0.3 m, a row home rooftop is textured, material-heavy, and geometrically regular in ways that resemble natural-image content more than satellite content. ImageNet's filters have learned those textures from 1.3 million photographs; Clay has not.

## Version 4: back to Clay, this time by the book

Before we declared Clay unsuitable for the task, we wanted to rule out a specific alternative explanation. Clay's official documentation does not prescribe neural fine-tuning for classification. Its [finetune-on-embeddings tutorial](https://clay-foundation.github.io/model/finetune/finetune-on-embeddings.html) runs Clay as a pure feature extractor, caches CLS tokens per chip, and trains a scikit-learn RandomForest on those cached vectors. The tutorial reaches 90 percent accuracy on a marina-detection task with 216 samples. If v2 had failed because we deviated from Clay's documented usage pattern, v4 would catch it.

We followed the recipe exactly. `ClayMAEModule.load_from_checkpoint` with the prescribed `model_size="large"`, `dolls=[16,32,64,128,256,768,1024]`, LINZ conditioning, zero time and latitude/longitude vectors as the docs sanction. The classifier was a 500-tree RandomForest with `class_weight="balanced"`. We went further than the tutorial in places, using a 3072-dimensional readout that concatenated CLS with mean-pooled and max-pooled patch tokens, and averaging over five rotation-and-flip test-time augmentations. We also compared against a linear probe and an isotonic-calibrated RF to rule out the classifier family as a confound.

At the default 0.5 probability threshold, the classifier predicts zero distressed parcels in the balanced validation set.

![Validation confusion matrix at threshold 0.5 — all 1,437 distressed val parcels are classified as stable.](validation_confusion_matrix.png)

At the F1-tuned threshold of 0.23, applied to the natural-distribution holdout, it predicts distressed on every parcel — 54,728 positives, zero negatives, recall 1.0, precision 0.014.

![Natural-distribution confusion matrix at tuned threshold 0.23 — the classifier flips to flagging every parcel in the city.](nat_distribution_confusion_matrix.png)

There is no threshold between those two collapses that yields a usable classifier. This is not a tuning problem, and it is not a classifier-capacity problem. It is a statement about the features themselves.

## What the embeddings actually look like

Two diagnostics from the v4 run make the underlying geometry visible. A t-SNE projection of the validation CLS embeddings is the first:

![t-SNE of Clay v1.5 CLS embeddings on the validation set — stable and distressed points are uniformly intermixed.](projection.png)

Stable (blue) and distressed (red) are uniformly intermixed. There is no region of Clay's 1024-dimensional embedding space that concentrates distressed parcels. No classifier operating on these features — a Random Forest, a linear probe, a neural head, any of them — can separate the classes, because the separating geometry is not there to be learned.

The Random Forest's feature importances give the same story in a different register. In a feature space that carries real signal for a task, a tree ensemble typically concentrates its splits on a small number of informative dimensions, producing a sharp power-law falloff in the sorted importance distribution. What we see instead is a nearly flat spread across all 1,024 dimensions, ranging from about 0.0013 at the top to 0.0006 at the bottom.

![Sorted Clay embedding dimension importances from the RandomForest classifier — nearly uniform across 1,024 dimensions, with no informative subset standing out.](clayimportance.png)

This is the classifier telling us it cannot find any useful dimensions, so it is averaging across all of them and hoping for the best. Combined with the t-SNE, the picture is consistent: the features do not contain a separating signal, and any classifier trained on them is effectively fitting noise.

A third diagnostic is cosine nearest-neighbor retrieval. Given a known-distressed query chip, we pull the six closest chips in Clay's embedding space:

![Nearest neighbors in Clay embedding space — five of six neighbors at cosine distances of 0.04 are stable, despite strong visual similarity to the distressed query.](nnclay.png)

Five of the six neighbors are stable, at cosine distances around 0.04. The visual similarity is real — all are pale flat roofs with similar orientation and material — and that is exactly the point. Clay's features faithfully capture roof material and geometry. They do not capture condition. The distress label and the visual-similarity label are not the same thing in this feature space, which means they cannot be the same thing at the output of any classifier built on top.

A final view, the full natural-distribution holdout colored by predicted probability, shows the same story at city scale:

![Predicted distress probability across Philadelphia's natural-distribution holdout, with classifier outcomes at threshold 0.23.](preds_naturaldistribution.png)

Predicted probabilities cluster in a narrow band between roughly 0.28 and 0.43 across the entire city. There is no high-confidence tail on either end, which is precisely why any threshold either catches everyone or catches no one.

## Two claims that survive all three experiments

Taken together, the three versions support one strong claim and one more tentative one.

The strong claim is about pretraining distance. For 0.3 m single-date aerial RGB, a satellite-foundation-model's core strength — scale-aware Vision Transformer training across sensors from 0.5 m LINZ through 10 m Sentinel-2 — is the wrong prior. At this resolution, the content of a chip is individual shingles and gutter lines, not urban morphology, and Clay's pretraining distribution never sees imagery where those are the dominant features. Three different classifier heads on Clay features (MLP with a frozen backbone, linear probe, calibrated Random Forest) all fail in structurally similar ways on the same data, while a general-purpose ImageNet backbone fine-tuned on the same chips produces a working classifier. Pretraining distance, not head architecture and not classifier capacity, is the bottleneck. No amount of downstream engineering bridges it.

The more tentative claim is about the task itself. The distress label is partially unobservable from directly above. L&I violations include foundation cracks, interior water damage, failed egress, and facade conditions that never rise above the roofline. The v4 nearest-neighbor analysis puts a ceiling on how much any purely overhead-RGB classifier can achieve: Clay reliably retrieves visually similar parcels, and those parcels carry mixed labels, because the label is not fully determined by what the imagery shows. This cap applies to v3 as well. Even our working classifier is being asked, some of the time, to predict something the pixels do not contain.

## Does temporal differencing rescue the task?

The obvious next move, once single-date features fail, is to add a temporal signal. A building that collapsed between 2024 and 2025 should look different in the two flights — a new tarp, fresh debris, a hole where a roof used to be. We computed four chip-level change features between the 2024 and 2025 aerial mosaics: mean absolute difference and 95th-percentile absolute difference over pixel values, structural similarity (SSIM) between the two dates, and `dark_new_frac`, the fraction of pixels that appeared newly dark in 2025 (a rough proxy for holes and shadows from collapsed structures).

The distributions, split by distress label, look like this:

![2024 to 2025 change feature distributions for stable (blue) and distressed (red) parcels. All four features show nearly complete overlap between the two classes.](change.png)

Stable and distressed are nearly indistinguishable on all four features. There is a small shift in SSIM (distressed parcels have slightly lower structural similarity between years, as we might hope) and a small tail in `dark_new_frac` that leans distressed, but nothing that approaches separability. The cleanest test of these features is to put them in a precision-recall curve against the Clay embeddings alone and against a combined Clay + change model:

![Validation precision-recall curves for Clay features only, change features only, and combined. All three hover just above the balanced-prior baseline of 0.33.](val_comp.png)

All three AP values sit within a few points of the balanced prior of 0.33 — Clay alone at 0.340, Change alone at 0.384, Combined at 0.362. The change features edge out Clay on average precision, which tells us they do contain marginally more signal than the frozen embedding, but none of the three approaches carries enough information to lift precision meaningfully above the prior at any useful recall.

This is a harder negative result than the single-date one, because it closes off the most obvious complementary signal. Temporal differencing on 0.3 m RGB is not the missing piece. The change features capture real pixel-level variation between years, but that variation is dominated by lighting, shadow angle, tree canopy seasonal effects, and vehicle placement — not by structural condition changes at a rate that differs between stable and distressed parcels.

## What we would ship, and what we would build next

If this were a deliverable rather than a class project, we would ship v3 as the working baseline. It is the only one of the three that produces a coherent natural-prior operating point, and it establishes a floor for future work.

With temporal differencing on RGB now in the evidence pile as a third failed approach, the remaining extensions have to reach further. Oblique imagery — Pictometry or equivalent — is the highest-value next step, because facade damage is where most Unsafe violations actually live, and oblique views capture it directly rather than trying to infer it from a top-down silhouette. Pairing each parcel with the nearest Mapillary or Google Street View image would serve the same purpose at lower acquisition cost. Non-imagery features that the L&I label genuinely depends on — parcel age, last inspection date, permit history, adjacency to previously distressed parcels — would let a classifier exploit the reactive structure of the label set instead of fighting it. And DEM differencing, rather than RGB differencing, would capture roof collapses as genuine elevation changes rather than as pixel-value shifts confounded by illumination.

We would also treat "Clay does not work here" as a publishable finding in its own right, rather than an engineering setback. The transfer limits of Earth-observation foundation models at sub-meter scales are not well characterized in the current literature, and a clean three-way comparison on a real urban task — with diagnostics showing where the feature geometry fails and a follow-up experiment showing that the obvious complementary signal also fails — is the kind of negative result that saves other groups from repeating the same path.

The t-SNE plot, the nearly flat importance spectrum, and the overlapping change-feature histograms are, together, the real output of the project. They answered the question we actually came to ask. The answer was not the one we expected, but it was the one that specifies what to try next.

---

*Code, embeddings, and the full evaluation notebooks are available in the project repository. Thanks to Nissim Lebovits and Guray Erus for the feedback throughout the semester.*
