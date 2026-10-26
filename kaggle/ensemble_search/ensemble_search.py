"""Ensemble search including v7, plus a free TTA measurement. GPU, ~15 min.

The config batch searched subsets of {A,B,C,D} and found A+C at gold 0.7347,
just under v7's 0.7353, and concluded ensembling doesn't help. But **v7 was
never in the pool**: the batch run had no v7 validation predictions to average,
so the search could only combine models that were each individually weaker than
v7. Averaging the two weakest-but-decent members recovered almost the whole gap,
which is the signature of an ensemble that wants its strongest member added.

This computes v7's predictions on the *identical* split the batch used (same
rng seed, same code path) and searches all 2^5-1 subsets of {v7,A,B,C,D}.

At the same time, and for free, it measures **TTA** (horizontal flip). Never
tried, and cited by the 0.809 public baseline. Note the prior is not obviously
positive here: geometry normalization already flips every knee to the same side,
so an hflip shows the encoder a mirrored knee it has never seen in training.
That is a reason to measure it, not to assume either sign.

Caveat carried into the reading of the output: gold-58 has +/-0.05 noise on the
macro. Searching 31 subsets against 58 studies will find a winner by noise
alone. Prefer the simplest combination whose margin is real, and treat the
leaderboard as the arbiter.

Also saves every model's val predictions to an npz, so any future ensemble
question is answerable locally with no Kaggle round-trip.
"""
import os
import subprocess
import sys
import time
from itertools import combinations

print("installing torch 2.6.0+cu118 (sm_60 kernels for the P100)", flush=True)
t0 = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.6.0", "torchvision==0.21.0",
                    "--index-url", "https://download.pytorch.org/whl/cu118"],
                   capture_output=True, text=True)
print(f"pip rc={r.returncode} in {time.time()-t0:.0f}s", flush=True)
if r.returncode != 0:
    print((r.stdout + r.stderr)[-1200:])
    sys.exit("install failed")

import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

assert torch.cuda.is_available(), "no CUDA after install"
dev = "cuda"
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)

BATCH_STUDIES = 4
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

print("input tree:", flush=True)
for _p in sorted(glob.glob("/kaggle/input/*")):
    for _q in sorted(glob.glob(_p + "/*"))[:12]:
        print("  ", _q, flush=True)

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
assert _npy and _idx, f"cache not found: {_npy} {_idx}"
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])

# --- the five checkpoints -----------------------------------------------------
# v7 arrives as dataset rsna-knee-finetuned (finetuned.pt); A-D as the batch
# kernel's output (model_<name>.pt).
CKPTS = {}
_v7 = glob.glob("/kaggle/input/**/finetuned.pt", recursive=True)
if _v7:
    CKPTS["v7"] = _v7[0]
for p in sorted(glob.glob("/kaggle/input/**/model_*.pt", recursive=True)):
    CKPTS[os.path.basename(p)[len("model_"):-len(".pt")]] = p
print("\ncheckpoints found:", flush=True)
for k, v in CKPTS.items():
    print(f"  {k:<12} {v}", flush=True)
assert "v7" in CKPTS, "v7 checkpoint missing; attach catejsj/rsna-knee-finetuned"
assert len(CKPTS) >= 2, "need at least two models to ensemble"

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
print(f"\ncache {X.shape}", flush=True)

lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
G = lab.loc[ids, LABELS].values.astype(np.float32)
train_csv = pd.read_csv(os.path.join(COMP, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)
gold_mask = np.array([u in gold_ids for u in ids])
Y_gold = np.zeros((len(ids), len(LABELS)), dtype=np.int64)
Y_gold[gold_mask] = gold_df.loc[ids[gold_mask], LABELS].values.astype(int)

# identical to finetune.py and finetune_batch.py; do not change, the whole
# point is that these predictions are comparable to the batch's
rng = np.random.default_rng(0)
order = rng.permutation(len(ids))
val = np.unique(np.concatenate([np.where(gold_mask)[0],
                                order[:400][~gold_mask[order[:400]]]]))
print(f"val {len(val)} incl. {gold_mask.sum()} gold", flush=True)


class Studies(Dataset):
    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        return torch.from_numpy(np.asarray(X[self.index[k]]))


class AttnHead(nn.Module):
    """Must match the training definition exactly."""

    def __init__(self, dim, n_out, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(n_out, dim) * dim ** -0.5)
        self.value = nn.Linear(dim, hidden)
        self.out = nn.Parameter(torch.zeros(n_out, hidden))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, emb):
        e = self.norm(emb)
        attn = torch.einsum("bsd,cd->bcs", e, self.query) * e.shape[-1] ** -0.5
        pooled = torch.einsum("bcs,bsh->bch", attn.softmax(-1), self.value(e))
        return (pooled * self.out).sum(-1) + self.bias


from transformers import AutoModel
ROOT = [os.path.dirname(p) for p in
        glob.glob("/kaggle/input/**/config.json", recursive=True)
        if "dinov2" in p.lower()][0]


def gold_auc(preds):
    """Macro AUC over the 58 radiologist-labeled studies inside val."""
    gm = gold_mask[val]
    out = []
    for j in range(len(LABELS)):
        yg = Y_gold[val][gm, j]
        if len(np.unique(yg)) == 2:
            out.append(roc_auc_score(yg, preds[gm, j]))
    return float(np.mean(out))


def gold_auc_per_label(preds):
    gm = gold_mask[val]
    out = {}
    for j, name in enumerate(LABELS):
        yg = Y_gold[val][gm, j]
        if len(np.unique(yg)) == 2:
            out[name] = (roc_auc_score(yg, preds[gm, j]), int(yg.sum()))
    return out


@torch.no_grad()
def predict(path):
    """-> (plain, hflip) prediction arrays over val."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    backbone = AutoModel.from_pretrained(ROOT)
    backbone.load_state_dict(ck["backbone"])
    head = AttnHead(backbone.config.hidden_size, len(LABELS))
    head.load_state_dict(ck["head"])
    backbone, head = backbone.to(dev).eval(), head.to(dev).eval()

    def fwd(imgs, flip):
        b, s, h, w = imgs.shape
        x = imgs.reshape(b * s, 1, h, w).float().div_(255.)
        if flip:
            x = torch.flip(x, dims=[-1])
        x = x.repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        e = backbone(pixel_values=x).last_hidden_state[:, 0]
        return torch.sigmoid(head(e.reshape(b, s, -1)))

    outs = []
    for flip in (False, True):
        preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
        vl = DataLoader(Studies(val), batch_size=BATCH_STUDIES, shuffle=False,
                        num_workers=2)
        at = 0
        for imgs in vl:
            with torch.amp.autocast("cuda"):
                o = fwd(imgs.to(dev, non_blocking=True), flip)
            preds[at:at + len(o)] = o.float().cpu().numpy()
            at += len(o)
        outs.append(preds)
    del backbone, head
    torch.cuda.empty_cache()
    return outs[0], outs[1]


# --- 1. per-model predictions -------------------------------------------------
print("\n" + "=" * 66)
print(f"{'model':<12}{'plain':>10}{'hflip':>10}{'avg (TTA)':>12}{'time':>8}")
print("=" * 66)
plain, flipped = {}, {}
for name, path in CKPTS.items():
    t = time.time()
    try:
        p, f = predict(path)
        plain[name], flipped[name] = p, f
        print(f"{name:<12}{gold_auc(p):>10.4f}{gold_auc(f):>10.4f}"
              f"{gold_auc((p + f) / 2):>12.4f}{(time.time()-t)/60:>7.1f}m",
              flush=True)
    except Exception as e:
        print(f"{name:<12} FAILED: {str(e)[:70]}", flush=True)

names = list(plain)

# --- 2. ensemble search, with and without TTA ---------------------------------
for tta in (False, True):
    src = {k: (plain[k] + flipped[k]) / 2 for k in names} if tta else plain
    print("\n" + "=" * 66)
    print(f"ENSEMBLES {'WITH TTA' if tta else 'no TTA'} "
          f"(v7 alone = {gold_auc(src['v7']):.4f})")
    print("=" * 66)
    scored = []
    for k in range(1, len(names) + 1):
        for combo in combinations(names, k):
            scored.append((gold_auc(np.mean([src[c] for c in combo], axis=0)),
                           combo))
    scored.sort(reverse=True)
    for g, combo in scored[:10]:
        print(f"  {'+'.join(combo):<44}{g:>8.4f}", flush=True)
    print(f"  ... {len(scored)-10} more combinations")
    if not tta:
        best_plain = scored[0]
    else:
        best_tta = scored[0]

print("\n" + "=" * 66)
print(f"best without TTA: {'+'.join(best_plain[1]):<34}{best_plain[0]:.4f}")
print(f"best with    TTA: {'+'.join(best_tta[1]):<34}{best_tta[0]:.4f}")
print(f"v7 alone, no TTA: {'':<34}{gold_auc(plain['v7']):.4f}")
print("=" * 66)
print("gold-58 noise is +/-0.05 on the macro; a margin under that is not a")
print("result. 31 combinations searched against 58 studies WILL produce a")
print("winner by chance; prefer the simplest one whose margin is real.")

winner = best_tta if best_tta[0] > best_plain[0] else best_plain
use_tta = best_tta[0] > best_plain[0]
src = {k: (plain[k] + flipped[k]) / 2 for k in names} if use_tta else plain
final = np.mean([src[c] for c in winner[1]], axis=0)

print(f"\nper-label gold for {'+'.join(winner[1])}"
      f"{' +TTA' if use_tta else ''} vs v7 alone:")
print(f"  {'label':<20}{'winner':>9}{'v7':>9}{'delta':>9}{'n_pos':>7}")
w_pl, v_pl = gold_auc_per_label(final), gold_auc_per_label(plain["v7"])
for name in LABELS:
    if name in w_pl:
        a, n = w_pl[name]
        b, _ = v_pl[name]
        print(f"  {name:<20}{a:>9.4f}{b:>9.4f}{a-b:>+9.4f}{n:>7}")

np.savez_compressed("/kaggle/working/val_predictions.npz",
                    val_index=val, study_ids=ids[val],
                    gold_mask=gold_mask[val], Y_gold=Y_gold[val],
                    names=np.array(names),
                    plain=np.stack([plain[k] for k in names]),
                    flipped=np.stack([flipped[k] for k in names]))
np.savez_compressed("/kaggle/working/ensemble_choice_v2.npz",
                    members=np.array(winner[1]), gold=winner[0],
                    tta=use_tta, all_names=np.array(names))
print("\nsaved val_predictions.npz (all models, both flips) + "
      "ensemble_choice_v2.npz\nDONE", flush=True)
