"""EfficientNet-B3 validation predictions. GPU, ~3 min.

gold 0.7760, well below v10 (0.8438) and v14 (0.8508). That alone does not
settle it -- v11-base is worse than small on the leaderboard and still lifted the
ensemble by +0.010, because what a member contributes is decided by how
differently it fails, not how well it scores.

But 0.776 is a 0.075 deficit, ten times base's 0.007, and this model also emitted
0.2-0.4% non-finite predictions per epoch during training (EfficientNet + fp16 on
sm_60, contained by fp32 pooling and gradient clipping but not eliminated). Any
non-finite value is replaced with a neutral 0.5 here so a handful cannot poison
an average.
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
CKPT = glob.glob("/kaggle/input/**/finetuned_effnet.pt", recursive=True)[0]

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
assert X.shape[-1] == 224, f"expected the 224px cache, got {X.shape[-1]}px"
print(f"cache {X.shape}", flush=True)

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


ck = torch.load(CKPT, map_location="cpu", weights_only=False)
assert ck.get("arch") == "efficientnet_b3", f"unexpected arch {ck.get('arch')}"
# Guard against a stale dataset version. `kaggle datasets version` is async and a
# kernel launched too soon silently attaches the PREVIOUS version -- no error,
# just a plausible wrong number. The fp32 v3 model scores 0.8046; the fp16 v2 it
# replaced scored 0.7760.
_g = float(ck.get("gold", -1))
assert _g > 0.79, (
    f"checkpoint reports gold {_g:.4f}, expected ~0.8046 (fp32 v3). This is the "
    f"stale fp16 checkpoint -- the dataset version had not propagated yet.")
print(f"checkpoint gold {_g:.4f} -- fp32 v3 confirmed", flush=True)
DIM = int(ck.get("dim", 1536))
net = torchvision.models.efficientnet_b3(weights=None)
backbone = net.features
backbone.load_state_dict(ck["backbone"], strict=True)
head = AttnHead(DIM, len(LABELS))
head.load_state_dict(ck["head"], strict=True)
backbone, head = backbone.to(dev).eval(), head.to(dev).eval()
print(f"loaded, val gold recorded at training: {ck.get('gold', float('nan')):.4f}",
      flush=True)


@torch.no_grad()
def run():
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val), batch_size=2, shuffle=False, num_workers=2)
    at = 0
    for imgs in vl:
        b, s, h, w = imgs.shape
        x = imgs.to(dev, non_blocking=True).reshape(b * s, 1, h, w)
        x = x.float().div_(255.).repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
        with torch.amp.autocast("cuda"):
            f = backbone(x).float().mean(dim=(2, 3))
            o = torch.sigmoid(head(f.reshape(b, s, -1)))
        preds[at:at + len(o)] = o.float().cpu().numpy()
        at += len(o)
    return preds


t = time.time()
P = run()
# this model emitted 0.2-0.4% non-finite outputs during training; substitute the
# neutral 0.5 so a handful of bad voxels cannot poison an ensemble average
n_bad = int((~np.isfinite(P)).sum())
if n_bad:
    print(f"non-finite predictions: {n_bad}/{P.size} -> replaced with 0.5",
          flush=True)
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)
gm = gold_mask[val]
out = []
for j in range(len(LABELS)):
    y = Y_gold[val][gm, j]
    if len(np.unique(y)) == 2:
        out.append(roc_auc_score(y, P[gm, j]))
print(f"effnet gold {np.mean(out):.4f}  ({(time.time()-t)/60:.1f}m)", flush=True)

np.savez_compressed("/kaggle/working/val_predictions_effnet.npz",
                    val_index=val, names=np.array(["v15_effnet"]),
                    plain=np.stack([P]))
print("saved val_predictions_effnet.npz\nDONE", flush=True)
