"""RadImageNet ResNet-50 validation predictions. GPU, ~3 min.

Individual gold is not the question. 336px scored 0.8533 -- better than v10's
0.8438 -- and added +0.0004 to the ensemble, because it is still DINOv2-small
and makes the same mistakes. What matters is whether a CNN pretrained on
radiology fails *differently* from a ViT pretrained on photographs.

Predictions only; the search runs locally against the cached DINOv2 predictions,
which is safe because every training kernel builds `val` from the same rng(0)
permutation over the same study order.
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
CKPT = glob.glob("/kaggle/input/**/finetuned_radimagenet.pt", recursive=True)[0]

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
assert ck.get("arch") == "resnet50_radimagenet", f"unexpected arch {ck.get('arch')}"
net = torchvision.models.resnet50(weights=None)
backbone = nn.Sequential(*list(net.children())[:-2])
backbone.load_state_dict(ck["backbone"], strict=True)
head = AttnHead(2048, len(LABELS))
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
            f = backbone(x).mean(dim=(2, 3))
            o = torch.sigmoid(head(f.reshape(b, s, -1)))
        preds[at:at + len(o)] = o.float().cpu().numpy()
        at += len(o)
    return preds


t = time.time()
P = run()
gm = gold_mask[val]
out = []
for j in range(len(LABELS)):
    y = Y_gold[val][gm, j]
    if len(np.unique(y)) == 2:
        out.append(roc_auc_score(y, P[gm, j]))
print(f"radimagenet gold {np.mean(out):.4f}  ({(time.time()-t)/60:.1f}m)", flush=True)

np.savez_compressed("/kaggle/working/val_predictions_radimg.npz",
                    val_index=val, names=np.array(["v14_radimg"]),
                    plain=np.stack([P]))
print("saved val_predictions_radimg.npz\nDONE", flush=True)
