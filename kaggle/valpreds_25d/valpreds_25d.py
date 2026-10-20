"""Validation predictions for the 2.5D model. GPU, ~3 min.

gold 0.8367 against v17B's 0.8654 -- the identical model with only the input
channels changed, so -0.029 is attributable to 2.5D alone.

That fails the competitiveness half of the ensemble rule (helpful members have
sat within ~0.01 of the best). But being fed differently is a genuinely new axis
of decorrelation, and this project has been wrong before about what a weak model
contributes -- v11-base is worse than v10-small on the leaderboard and still
lifted the ensemble by +0.010. Measuring is cheap; assuming has cost more.

The 2.5D channel construction here must match training exactly. If it reverted to
`.repeat(1, 3, 1, 1)` the weights would be fed an input distribution they never
saw and the predictions would be noise that still looks like probabilities.
"""
import os
import subprocess
import sys
import time

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
import torchvision
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

assert torch.cuda.is_available(), "no CUDA after install"
dev = "cuda"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])

CKPTS = {}
for tag, fname in [("v18_25d", "finetuned_radimg_25d.pt")]:
    f = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
    assert f, f"missing {fname}"
    CKPTS[tag] = f[0]
print("checkpoints:", list(CKPTS), flush=True)

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
assert X.shape[-1] == 224, f"expected 224px cache, got {X.shape[-1]}px"

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
gm = gold_mask[val]
print(f"val {len(val)} incl. {gm.sum()} gold", flush=True)


class Studies(Dataset):
    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        return torch.from_numpy(np.array(X[self.index[k]]))


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


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck.get("arch") == "resnet50_radimagenet", f"arch {ck.get('arch')}"
    # stale-version guard: A trained to 0.8585, B to 0.8653, the original v14 to
    # 0.8508. `kaggle datasets version` is async and a kernel launched too soon
    # silently attaches the previous version, with no error at all.
    g = float(ck.get("gold", -1))
    assert 0.83 < g < 0.84, (
        f"checkpoint reports gold {g:.4f}, expected ~0.8367 for the 2.5D "
        f"model -- wrong or stale checkpoint")
    assert ck.get("input") == "2.5d", f"input tag {ck.get('input')}"
    net = torchvision.models.resnet50(weights=None)
    b = nn.Sequential(*list(net.children())[:-2])
    b.load_state_dict(ck["backbone"], strict=True)
    h = AttnHead(2048, len(LABELS))
    h.load_state_dict(ck["head"], strict=True)
    print(f"  {os.path.basename(path)}: variant {ck.get('variant')}, "
          f"training gold {g:.4f}", flush=True)
    return b.to(dev).eval(), h.to(dev).eval()


@torch.no_grad()
def predict(backbone, head):
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val), batch_size=2, shuffle=False, num_workers=2)
    at = 0
    for imgs in vl:
        b, s, h, w = imgs.shape
        v = imgs.to(dev, non_blocking=True).reshape(b, 3, s // 3, h, w)
        v = v.float().div_(255.)
        prev = torch.cat([v[:, :, :1], v[:, :, :-1]], dim=2)
        nxt = torch.cat([v[:, :, 1:], v[:, :, -1:]], dim=2)
        x = torch.stack([prev, v, nxt], dim=3).reshape(b * s, 3, h, w)
        x = (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
        with torch.amp.autocast("cuda"):
            f = backbone(x).float().mean(dim=(2, 3))
            o = torch.sigmoid(head(f.reshape(b, s, -1)))
        preds[at:at + len(o)] = o.float().cpu().numpy()
        at += len(o)
    return preds


def gold_auc(p):
    out = []
    for j in range(len(LABELS)):
        y = Y_gold[val][gm, j]
        if len(np.unique(y)) == 2:
            out.append(roc_auc_score(y, p[gm, j]))
    return float(np.mean(out))


names, allp = [], []
for tag, path in CKPTS.items():
    backbone, head = load(path)
    t = time.time()
    P = predict(backbone, head)
    n_bad = int((~np.isfinite(P)).sum())
    if n_bad:
        print(f"    WARNING {n_bad}/{P.size} non-finite -> 0.5", flush=True)
        P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)
    print(f"    {tag} gold {gold_auc(P):.4f}  ({(time.time()-t)/60:.1f}m)",
          flush=True)
    names.append(tag)
    allp.append(P)
    del backbone, head
    torch.cuda.empty_cache()

assert len(allp) == len(CKPTS), f"collected {len(allp)} of {len(CKPTS)}"
np.savez_compressed("/kaggle/working/val_predictions_25d.npz",
                    val_index=val, names=np.array(names),
                    plain=np.stack(allp))
print(f"\nsaved val_predictions_25d.npz with {len(allp)} members\nDONE",
      flush=True)
