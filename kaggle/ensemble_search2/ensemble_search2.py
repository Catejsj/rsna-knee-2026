"""Ensemble search over the label-repaired models. GPU, ~10 min.

The first search settled the method: averaging predictions works, and it works
because the pool contained the strongest member. v7+A+C scored gold 0.7396 and
delivered **LB 0.800** against v7's 0.795 -- gold called it to 0.001.

Since then the labels were repaired and two much stronger models exist:

    v7   gold 0.7353   LB 0.795
    v8   gold 0.7655   LB 0.807   (rules_v4)
    v9   gold 0.7811   LB pending (rules_v5)

This searches every subset of {v7, v8, v9, A, C}. A and C are carried over
because the first search showed weak members still contribute when a strong one
anchors the average; they cost nothing to include in the search and can be
dropped if they do not earn a place.

Only v8 and v9 need computing -- v7, A and C were cached by the first search as
`val_predictions.npz`, on the identical split (rng seed 0, same code in
finetune.py / finetune_v4.py / finetune_v5.py, all of which build `val` the same
way). That makes every model's predictions directly comparable.

No TTA. It was measured in the first search and lost for all five models and all
31 combinations, because geometry normalization already flips every knee to the
same side.
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

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])

CACHED = glob.glob("/kaggle/input/**/val_predictions.npz", recursive=True)
assert CACHED, "val_predictions.npz not found -- attach catejsj/rsna-ensemble-search"
prev = np.load(CACHED[0], allow_pickle=True)
prev_names = [str(s) for s in prev["names"]]
print(f"cached models: {prev_names}", flush=True)

NEW = {}
for tag, fname in [("v8", "finetuned_v4.pt"), ("v9", "finetuned_v5.pt")]:
    f = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
    assert f, f"missing {fname}"
    NEW[tag] = f[0]
print(f"to compute: {list(NEW)}", flush=True)

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")

train_csv = pd.read_csv(os.path.join(COMP, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)
gold_mask = np.array([u in gold_ids for u in ids])
Y_gold = np.zeros((len(ids), len(LABELS)), dtype=np.int64)
Y_gold[gold_mask] = gold_df.loc[ids[gold_mask], LABELS].values.astype(int)

rng = np.random.default_rng(0)
order = rng.permutation(len(ids))
val = np.unique(np.concatenate([np.where(gold_mask)[0],
                                order[:400][~gold_mask[order[:400]]]]))
assert np.array_equal(val, prev["val_index"]), \
    "split drifted from the cached run -- predictions are not comparable"
print(f"val {len(val)} incl. {gold_mask.sum()} gold; split matches cache",
      flush=True)


class Studies(Dataset):
    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        return torch.from_numpy(np.asarray(X[self.index[k]]))


class AttnHead(nn.Module):
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
    gm = gold_mask[val]
    out = []
    for j in range(len(LABELS)):
        yg = Y_gold[val][gm, j]
        if len(np.unique(yg)) == 2:
            out.append(roc_auc_score(yg, preds[gm, j]))
    return float(np.mean(out))


@torch.no_grad()
def predict(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    backbone = AutoModel.from_pretrained(ROOT)
    backbone.load_state_dict(ck["backbone"])
    head = AttnHead(backbone.config.hidden_size, len(LABELS))
    head.load_state_dict(ck["head"])
    backbone, head = backbone.to(dev).eval(), head.to(dev).eval()
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val), batch_size=BATCH_STUDIES, shuffle=False,
                    num_workers=2)
    at = 0
    for imgs in vl:
        b, s, h, w = imgs.shape
        x = imgs.to(dev, non_blocking=True).reshape(b * s, 1, h, w)
        x = x.float().div_(255.).repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
        with torch.amp.autocast("cuda"):
            e = backbone(pixel_values=x).last_hidden_state[:, 0]
            o = torch.sigmoid(head(e.reshape(b, s, -1)))
        preds[at:at + len(o)] = o.float().cpu().numpy()
        at += len(o)
    del backbone, head
    torch.cuda.empty_cache()
    return preds


# keep v7 and the two batch members that earned a place last time
P = {n: prev["plain"][prev_names.index(n)]
     for n in ("v7", "A_e4_u4", "C_e4_u6") if n in prev_names}
for tag, path in NEW.items():
    t = time.time()
    P[tag] = predict(path)
    print(f"  computed {tag}: gold {gold_auc(P[tag]):.4f} "
          f"({(time.time()-t)/60:.1f}m)", flush=True)

print("\n" + "=" * 62)
print(f"{'member':<14}{'gold':>9}")
print("=" * 62)
for n in P:
    print(f"{n:<14}{gold_auc(P[n]):>9.4f}", flush=True)

names = list(P)
scored = []
for k in range(1, len(names) + 1):
    for combo in combinations(names, k):
        scored.append((gold_auc(np.mean([P[c] for c in combo], axis=0)), combo))
scored.sort(reverse=True)

print("\n" + "=" * 62)
print("TOP ENSEMBLES")
print("=" * 62)
for g, combo in scored[:14]:
    print(f"  {'+'.join(combo):<44}{g:>8.4f}", flush=True)

best_g, best = scored[0]
solo = max((gold_auc(P[n]), n) for n in names)
print(f"\nBEST      {'+'.join(best)}  {best_g:.4f}")
print(f"BEST SOLO {solo[1]}  {solo[0]:.4f}")
print(f"margin    {best_g - solo[0]:+.4f}")
print("\ngold-58 is +/-0.05 on the macro and over-predicted v8 by 0.018, so")
print("treat the margin as directional. Prefer the simplest combination.")

np.savez_compressed("/kaggle/working/val_predictions_v2.npz",
                    val_index=val, names=np.array(names),
                    plain=np.stack([P[n] for n in names]))
np.savez_compressed("/kaggle/working/ensemble_choice_v3.npz",
                    members=np.array(best), gold=best_g)
print("\nsaved val_predictions_v2.npz + ensemble_choice_v3.npz\nDONE", flush=True)
