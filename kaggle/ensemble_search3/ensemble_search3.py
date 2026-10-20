"""Does a genuinely different architecture finally make ensembling pay? GPU.

Every ensemble result so far has been limited by correlation. v7+A+C helped
(+0.005 LB) because the members differed in schedule. v7+v8+v9 *hurt* -- 0.7762
against v8+v9's 0.7839 -- because once the anchor is strong, members sharing
architecture, data and split add nothing but their own noise.

DINOv2-base is the first member that differs in the encoder itself: 768-dim and
~86M parameters against small's 384-dim and ~22M. If architecture diversity is
what was missing, this is where it shows up.

    v10  small, LLM labels   gold 0.8438   LB 0.870
    v11  base,  LLM labels   gold 0.8541

Both rank and probability averaging are scored. Rank averaging lost on our
earlier models (-0.0016) and the stated reason was that it only pays when
members have different score scales -- which same-architecture members do not.
Two different encoders is exactly the case where that reasoning predicts it
should start working, so this is a real test of that explanation rather than a
repeat of a settled negative.

Backbone size is read from each checkpoint's head LayerNorm rather than assumed,
so a checkpoint can never be loaded into the wrong encoder.
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

CKPTS = {}
for tag, fname in [("v10_small", "finetuned_llm.pt"),
                   ("v11_base", "finetuned_llm_base.pt"),
                   ("v12_leehans", "finetuned_leehans.pt"),
                   ("v9_small", "finetuned_v5.pt")]:
    f = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
    if f:
        CKPTS[tag] = f[0]
    else:
        print(f"  note: {fname} not attached, skipping {tag}", flush=True)
assert len(CKPTS) >= 2, f"need two checkpoints, found {list(CKPTS)}"
print("checkpoints:", list(CKPTS), flush=True)

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
print(f"val {len(val)} incl. {gold_mask.sum()} gold", flush=True)


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
ROOTS = {}
for p in glob.glob("/kaggle/input/**/config.json", recursive=True):
    if "dinov2" not in p.lower():
        continue
    try:
        import json
        h = json.load(open(p)).get("hidden_size")
    except Exception:
        continue
    if h:
        ROOTS[int(h)] = os.path.dirname(p)
print(f"available encoders by hidden size: { {k: os.path.basename(v) for k, v in ROOTS.items()} }",
      flush=True)


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
    dim = int(ck["head"]["norm.weight"].shape[0])      # never assume the size
    assert dim in ROOTS, f"no encoder with hidden_size {dim}; have {list(ROOTS)}"
    backbone = AutoModel.from_pretrained(ROOTS[dim])
    backbone.load_state_dict(ck["backbone"])
    head = AttnHead(dim, len(LABELS))
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
    print(f"    {os.path.basename(path)}: dim {dim}", flush=True)
    return preds


P = {}
for tag, path in CKPTS.items():
    t = time.time()
    P[tag] = predict(path)
    print(f"  {tag:<12} gold {gold_auc(P[tag]):.4f}  ({(time.time()-t)/60:.1f}m)",
          flush=True)


def rank(p):
    return pd.DataFrame(p).rank(method="average", pct=True).to_numpy()


Pr = {n: rank(P[n]) for n in P}
names = list(P)

print("\n" + "=" * 66)
print(f"{'combination':<30}{'prob-avg':>11}{'rank-avg':>11}{'delta':>10}")
print("=" * 66)
rows = []
for k in range(1, len(names) + 1):
    for c in combinations(names, k):
        a = gold_auc(np.mean([P[x] for x in c], axis=0))
        b = gold_auc(np.mean([Pr[x] for x in c], axis=0))
        rows.append((max(a, b), a, b, c))
rows.sort(reverse=True)
for _, a, b, c in rows:
    print(f"{'+'.join(c):<30}{a:>11.4f}{b:>11.4f}{b-a:>+10.4f}", flush=True)

best_score, ba, bb, bc = rows[0]
solo = max((gold_auc(P[n]), n) for n in names)
print(f"\nBEST      {'+'.join(bc)}  {best_score:.4f} "
      f"({'rank' if bb > ba else 'prob'}-avg)")
print(f"BEST SOLO {solo[1]}  {solo[0]:.4f}")
print(f"margin    {best_score - solo[0]:+.4f}")
print("\nIf rank-avg wins here but lost on the same-architecture models, the")
print("scale explanation holds. If it loses again, that explanation was wrong.")

np.savez_compressed("/kaggle/working/val_predictions_v3.npz",
                    val_index=val, names=np.array(names),
                    plain=np.stack([P[n] for n in names]))
print("\nsaved val_predictions_v3.npz\nDONE", flush=True)
