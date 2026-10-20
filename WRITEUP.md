# Knee MRI from report-derived labels: what moved the score, and what did not

**RSNA 2026 Knee Abnormality Detection** — an independent pipeline, built from
scratch rather than forked, scoring **0.910 public LB**.

This writes up roughly 30 measured experiments: what moved the score, what did
not, and the measurement discipline that made the difference between the two
legible. The negative results are reported with the same detail as the
positive ones, because most of them are reusable.

> Written before the private leaderboard was revealed; all scores below are
> public-LB or gold-58.

---

## 1. The task, and the fact that shaped everything

Predict 12 findings per knee MRI study (ACL, MCL, medial/lateral meniscus,
medial/lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, contusion,
fracture). Metric: macro-averaged ROC-AUC. Code competition, ≤9 h, offline.

**4,407 training studies. 58 carry labels.** The other 4,349 have only a
free-text radiology report, in about 12 languages. Test studies have no
reports.

So every model here trains on labels mined from text, and is scored on labels
that — as far as can be told — come from the images:

- identical phrases ("small joint effusion") carry opposite gold labels in
  different studies;
- for Synovitis, 56% of gold positives are not mentioned in their report under
  any vocabulary (checked with a deliberately over-generous term bank in all 12
  languages).

## 2. Measurement: choose with gold-58, confirm on the leaderboard

The single most important decision in the project was which number to trust.

**Cross-validation against our own derived labels was wrong three times** —
twice in magnitude, once in sign:

| change | derived-label CV | gold-58 | public LB |
|---|---|---|---|
| 3 planes instead of 1 | **+0.061** | +0.000 | +0.005 |
| geometry normalisation | +0.001 | — | **+0.036** |
| graded (soft) targets | **−0.025** | +0.028 | **+0.029** |
| fine-tuning the encoder | — | +0.054 | +0.053 |

The reason is structural. Derived CV scores agreement with the extracted labels,
so it rewards fitting the extractor — including its systematic errors, which
repeat identically across folds and are therefore invisible to CV.

**Rule used from then on:** select on the 58 gold studies, confirm on the LB,
ignore derived CV. Every checkpoint is also selected on gold, not on loss —
several runs peaked on gold at epoch 3–4 while loss and derived CV kept
improving through epoch 8.

**The noise floor, measured.** The same recipe with a fixed seed reproduced
gold 0.8619 to four decimals on two different GPU types. Without a fixed seed
it scored 0.8700. So **±0.008 is pure run-to-run variance**, and in practice a
gold delta under ~0.015 carries no information. Gold also got one direction
wrong outright (DINOv2-base vs small), which is what set that threshold.

## 3. The pipeline

```
DICOM -> geometry normalisation -> 3 planes x 12 slices @224px
      -> encoder -> per-finding attention head -> 12 probabilities
                                 -> probability-average across members
```

**Geometry normalisation (+0.036 LB).**
1. Order slices by physical position along the slice normal
   (`ImagePositionPatient` · normal of `ImageOrientationPatient`); file order is
   arbitrary.
2. Crop a fixed **160 mm** of anatomy before resizing. Acquired field of view
   ranges 70–320 mm, so a fixed pixel resize shows the same knee at very
   different scales.
3. Flip every knee to face the same way. The laterality tag exists on only
   49.7% of studies; the rest are inferred from the sign of the image-centre x
   coordinate in patient space (97.1% agreement where the tag exists). Flipping
   *only* the tagged half made results worse: two conventions are worse than
   none.

**Labels.** A multilingual rules extractor emitting graded severity
(0.90 / 0.65 / 0.25 / 0.08), later replaced by public LLM-derived labels.
Measuring extractor coverage *per language rather than per script* exposed that
Turkish — 26.5% of the corpus — had almost no coverage: it is Latin-script, so
it had been averaged into the "Latin" bucket. Fixing it (term banks, the
dotless ı, and verb-final negation) was worth +0.030 gold.

**Head.** One learned query per finding attends over the study's 36 slice
embeddings. Different findings live on different slices and planes; a mean
dilutes each with 35 others.

**Ensemble.** Members are chosen by **correlation, not accuracy** — with a
competitiveness floor:

| member | gold | correlation with ensemble | effect |
|---|---|---|---|
| DINOv2-small @336px | 0.8533 | 0.934 | +0.0004 |
| RadImageNet ResNet-50 | 0.8508 | 0.856 | **+0.016 LB** |
| EfficientNet-B3 | 0.8082 | 0.847 | −0.003 |
| frozen MedImageInsight | 0.7986 | 0.79 | −0.006 |

The higher-resolution DINOv2 was individually *better* and added nothing; a
radiology-pretrained ResNet was individually worse and gave the largest single
ensemble gain. But decorrelation without competitiveness hurts — a member must
land within ~0.01 of the best.

The ensemble was selected by an exhaustive search over all 41,225 subsets of 17
trained members **under a runtime model** (the scored rerun is capped at 9 h).
The cost-blind optimum scored +0.0002 over the shipped ensemble and would have
taken 12.75 h — i.e. it would have scored nothing.

## 4. Progression

| step | public LB |
|---|---|
| 64px thumbnails, 5 slices | 0.605 |
| geometry normalisation | 0.669 |
| DINOv2 features | 0.713 |
| graded targets | 0.742 |
| fine-tuned DINOv2 + attention head | 0.795 |
| repaired multilingual labels | 0.827 |
| public LLM labels | 0.870 |
| + DINOv2-base, cross-architecture ensemble | 0.880 |
| + RadImageNet ResNet-50 | 0.896 |
| RadImageNet layer3+4 unfrozen, 4-model ensemble | 0.907 |
| +10% data (released non-gold holdouts) | 0.908 |
| 5-member ensemble without DINOv2-base | **0.910** |

## 5. What did not work — with numbers

All on gold-58 against a matched control unless marked LB. "Noise" = inside
the ±0.015 band.

| idea | result | why, as far as measured |
|---|---|---|
| 3 planes on raw pixels | +0.000 | planes matter for encoder features, not pixels |
| topology (Betti curves) | +0.005 | signal real but already in the encoder |
| persistent homology (ULBP landmarks) | 0.563 alone | a knee slice mixes tissues in one point cloud |
| Eikonal arrival-time channels | −0.055 | unchanged with conv1 unfrozen: genuinely less informative |
| 2.5D (neighbour slices as RGB) | −0.029 → −0.014 → −0.008 | see note below |
| 336px | +0.0004 | same model family, correlated errors |
| fold averaging | 0.8506 vs 0.8508 | folds correlate 0.94–0.97 |
| rank averaging | −0.0015 | probability averaging ships |
| TTA (horizontal flip) | negative | every knee is already flipped |
| pairwise ranking / AUC-surrogate loss | −0.005 hybrid, −0.024 pure | discards the graded targets |
| AUM label correction | −0.001 / −0.007 | flags rare positives, not report omissions |
| Dawid–Skene over 9 public teachers | 0.850 vs 0.893 best single | binarises; teachers' errors correlated |
| zero-shot open decision model as teacher | 0.52–0.60 | not a medical reader out of the box |
| frozen MedImageInsight member | 0.799, hurts ensemble | frozen features lose every time here |
| retrieval: kNN / slice matching / Optimal Transport / graph propagation | 0.67–0.71 vs 0.80 trained head | untrained similarity measures acquisition, not pathology |

Notes on the ones that teach something:

**2.5D, three attempts.** The first measurement had conv1 frozen; unfreezing it
halved the deficit, which is a general lesson — *when you replace a pretrained
model's input channels, unfreeze conv1 or you are measuring the frozen layer.*
The residual was then traced to the slices not being neighbours at all: the
loader subsamples to 32 files and the stacker takes 12 anchors, so the cached
"neighbours" were **6.60 mm** apart against a true gap of **3.30 mm**. With
genuinely contiguous triplets and conv1 trainable, it still came out −0.008. In
this pipeline (ResNet-50, 224px) it did not help. Public pipelines use 2.5D
successfully with a 384px CoAtNet — see §7.

**AUM label correction.** The hypothesis was that the image model would flag
findings the report omitted. The flag directions refuted it directly: 4,661
flags were "report positive, model negative" against 400 the other way; for
Synovitis, 427 against 7. Raw-margin AUM under imbalanced BCE identifies hard
rare positives, not label noise. (The margin was not prior-adjusted; a
prior-corrected version might flag differently.)

**Nine public label sources.** Every independent teacher scored ≤0.90 on gold,
and no combination (Dawid–Skene, mean, rank-mean) got past 0.896. One public
"label" dataset turned out to be the 58 gold labels re-uploaded — it scores a
perfect 1.0 on gold. Any public teacher should be checked for this before its
gold score means anything.

**Retrieval.** Four variants of "find the most similar training knee" all lost
to a trained readout by 0.09–0.13, and the finer the matching (slice-level,
Optimal Transport), the worse. Matching fixes alignment, not what *similar*
means: in an untrained embedding, nearest neighbours share scanner, plane and
contrast, not pathology. Image clustering had shown the same thing (clusters
recovered scan plane; pathology AMI ≤ 0.042). The remaining version —
retrieval in a space trained with a supervised contrastive loss
(`kaggle/finetune_supcon`) — was written and checked locally but its Kaggle run
failed to start, so it is **unmeasured**. It is included as code only.

## 6. Infrastructure findings that cost the most time

- **Submissions can run on GPU.** The rules allow "GPU Notebook ≤ 9 hours". This
  pipeline ran on CPU for two months because the old P100 rejected Kaggle's
  stock torch and an offline kernel cannot pip install. Internet-off GPU kernels
  now get a T4 that the stock torch supports. The same ensemble scored the
  identical 0.908 with predictions matching the CPU run to 5.4×10⁻⁷, in ≤1 h 39 m
  instead of ~7.5 h.
- **The 9 h cap applies to the scored rerun, not the 12 h CPU kernel limit.** A
  6-member ensemble estimated at 9.8 h scored nothing: the submission reads
  COMPLETE with an empty score.
- **Local benchmarks were 3.5× optimistic, four times.** Long jobs now run a
  throughput probe on Kaggle and refuse to start if the projection overruns.
- **Silent staleness:** `kaggle kernels output` does not overwrite existing
  files, and `kaggle datasets version` is asynchronous, so a kernel can attach
  the previous checkpoint. The fix that works is asserting on each checkpoint's
  own recorded gold score inside the kernel.
- A "`.pt`" checkpoint that was actually safetensors; a cache file named
  `slices_224_*` regardless of its real resolution. Shapes and headers are
  trustworthy; filenames are not.

## 7. Where this pipeline stops, and what the stronger public ones do

At writing, 5,207 teams; **1,016 sit at exactly 0.943**, which is a public
notebook, and about 1,600 at ≥0.942. Reading the strongest public pipelines
shows four differences from this one, none of which were tested here:

1. **More MRI sequences.** One series per *slot* — plane × contrast × fat
   suppression, six slots — with each finding's attention seeded toward the
   slots a radiologist reads. This pipeline uses one series per plane (three)
   and discards the rest.
2. **Denser slice sampling.** 62–94 slice windows per model against 12 per
   plane here.
3. **Stronger, higher-resolution backbones.** A 384px CoAtNet trained on
   contiguous 2.5D windows with a joint ROI crop, and top-epoch weight
   averaging.
4. **Very large ensembles of public weights,** made to fit in 9 h on 2×T4 by
   sharing identical encoder layers across members.

The gold-58 results above say the report labels top out near 0.90 on gold. The
leaderboard says models trained on similar labels reach 0.96 on test. Both can
be true — gold-58 is 58 enriched studies, not the test distribution — but the
practical reading is that this pipeline's gap is in the image side (sequences,
slice coverage, resolution, backbone), not in the labels. That is the most
useful thing this writeup can say to someone starting the same problem.

## 8. Reproducibility

Every experiment above is a Kaggle kernel with a fixed seed and a matched
control; each one asserts on its own inputs (cache shape, checkpoint gold,
label coverage) rather than trusting filenames. The kernels and local analysis
scripts are in this repository; competition data, derived labels and trained
weights are not, because the competition rules do not allow redistributing
them.

Final selections: the 5-member ensemble without DINOv2-base (0.910) and the
4-member ensemble with it (0.908), both run on GPU with predictions verified
against their CPU versions to within 6×10⁻⁷.
