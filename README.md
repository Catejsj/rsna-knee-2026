# RSNA 2026 Knee Abnormality Detection — an independent pipeline

Code for an entry in the Kaggle **RSNA 2026 Knee Abnormality Detection**
competition: predict 12 findings per knee MRI study from images, where only 58
of 4,407 training studies carry labels and the rest have only a free-text
radiology report in ~12 languages.

Built from scratch rather than forked. Best public leaderboard score **0.910**
(macro ROC-AUC).

**Read [`WRITEUP.md`](WRITEUP.md) first.** It documents ~30 measured experiments
— what moved the score, what did not, and why — including the negative results,
which are most of the reusable content.

## Layout

```
WRITEUP.md            the full write-up: method, measurements, negative results
src/                  local analysis: label extraction, scoring, ensemble search,
                      teacher comparison, retrieval experiments, Eikonal solver,
                      topology / persistent-homology descriptors
kaggle/<kernel>/      one directory per Kaggle kernel: <name>.py + kernel-metadata.json
  cache_*             decode DICOM once into uint8 slice caches
  finetune_*          training runs (each one a matched experiment with a fixed seed)
  valpreds_*          validation predictions consumed by the ensemble search
  submit_*            submission kernels (offline; CPU and GPU variants)
tools/                helper to ship src/ modules to Kaggle as a dataset
```

## What is not here, and why

- **Competition data and anything derived from it** — DICOMs, `train.csv`
  (which contains the reports), extracted label files, cached arrays, kernel
  outputs. The competition rules do not permit redistributing them.
- **Trained weights.** Large, and regenerable from the kernels.

## Reproducing

Everything runs as Kaggle kernels. The general pattern:

```bash
# local analysis (Python 3.11)
pip install -r requirements.txt
python src/extract_rules.py rules_v5        # reports -> 12 graded labels (needs competition data in data/)
python src/evaluate.py rules_v5             # score an extractor on the 58 gold studies
python src/ensemble_cost_search.py          # best ensemble that fits the 9 h cap

# Kaggle (edit the "id" in each kernel-metadata.json to your own username first)
python tools/sync_utility.py --push         # ship src/ modules as a private dataset
kaggle kernels push -p kaggle/cache_slices  # then finetune_*, valpreds_*, submit_*
```

Each kernel's docstring states what it tests, its matched control, and the
result it produced. Kernels assert on their own inputs (cache shape, the gold
score recorded inside each checkpoint, label coverage) rather than trusting file
names, because several silent staleness bugs were caught that way.

## Key results in one table

| | gold-58 | public LB |
|---|---|---|
| fine-tuned DINOv2 + per-finding attention head | 0.7353 | 0.795 |
| + public LLM labels | 0.8438 | 0.870 |
| + RadImageNet ResNet-50, cross-architecture ensemble | 0.8756 | 0.896 |
| 4-model ensemble (ens7) | 0.8861 | 0.908 |
| 5-model ensemble without DINOv2-base | 0.8838 | **0.910** |

## License

MIT for the code in this repository. Pretrained weights used by the kernels
(DINOv2, RadImageNet, MedImageInsight) and public label datasets keep their own
licenses.
