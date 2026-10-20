"""How much accuracy does cutting slices cost? GPU, ~15 min.

**The binding constraint is now runtime, not ideas.** The 6-member ensemble
scores gold 0.8802 against the shipped 0.8781 and cannot be run: it estimates
~9.8 h against a 9 h submission cap, and it timed out and scored nothing.
DINOv2-base alone costs 3 of those 9 hours.

Inference cost is very close to linear in slice count -- every model encodes
S slices per study and the attention head pools them. So if 8 slices per plane
costs little accuracy against the current 12, the whole budget stretches by 1.5x
and the 0.8802 ensemble becomes affordable, with room to spare.

**No retraining is required.** `n_slices` is read from the checkpoint and the
head pools over a variable number of slices, so a model trained on 12/plane can
be evaluated on 8 or 6 and it simply attends over fewer.

Caveat worth stating: this subsamples the cached 12-per-plane stack, whereas a
real reduced-slice submission would select k slices directly from the DICOM
series over the same physical band. Those are close but not identical -- the
subsample is drawn from an already-chosen set. Treat the numbers as a good first
read on the shape of the tradeoff, not as the exact score a k-slice submission
would post.

Measures every shipped member at 12 / 8 / 6 slices per plane, plus the ensemble
average, so the decision is made on the ensemble rather than on any one model.
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
PER_PLANE = [12, 8, 6]

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])

MEMBERS = {"v11_base": "finetuned_llm_base.pt",
           "v12_leehans": "finetuned_leehans.pt",
           "v14_radimg": "finetuned_radimagenet.pt",
           "v16_radf3": "finetuned_radimg_f3.pt"}
CKPTS = {}
for tag, fname in MEMBERS.items():
    f = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
    assert f, f"missing {fname}"
    CKPTS[tag] = f[0]
print("members:", list(CKPTS), flush=True)

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
assert X.shape[-1] == 224, f"expected 224px cache, got {X.shape[-1]}px"
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
gm = gold_mask[val]


def keep_indices(k):
    """k evenly spaced slices from each plane's block of 12, in cache order."""
    out = []
    for plane in range(3):
        sel = np.unique(np.linspace(0, 11, k).astype(int))
        out.extend((plane * 12 + sel).tolist())
    return np.array(out)


class Studies(Dataset):
    def __init__(self, index, keep):
        self.index, self.keep = index, keep

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        return torch.from_numpy(np.asarray(X[self.index[k]])[self.keep])


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
import json as _json
ROOTS = {}
for p in glob.glob("/kaggle/input/**/config.json", recursive=True):
    if "dinov2" not in p.lower():
        continue
    try:
        h = _json.load(open(p)).get("hidden_size")
    except Exception:
        continue
    if h:
        ROOTS[int(h)] = os.path.dirname(p)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    dim = int(ck["head"]["norm.weight"].shape[0])
    if ck.get("arch") == "resnet50_radimagenet":
        net = torchvision.models.resnet50(weights=None)
        seq = nn.Sequential(*list(net.children())[:-2])
        seq.load_state_dict(ck["backbone"], strict=True)
        b, kind = seq, "resnet"
    else:
        b = AutoModel.from_pretrained(ROOTS[dim])
        b.load_state_dict(ck["backbone"])
        kind = "dinov2"
    h = AttnHead(dim, len(LABELS))
    h.load_state_dict(ck["head"])
    return b.to(dev).eval(), h.to(dev).eval(), kind


def gold_auc(preds):
    out = []
    for j in range(len(LABELS)):
        y = Y_gold[val][gm, j]
        if len(np.unique(y)) == 2:
            out.append(roc_auc_score(y, preds[gm, j]))
    return float(np.mean(out))


@torch.no_grad()
def predict(backbone, head, kind, keep):
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val, keep), batch_size=2, shuffle=False, num_workers=2)
    at = 0
    for imgs in vl:
        b, s, h, w = imgs.shape
        x = imgs.to(dev, non_blocking=True).reshape(b * s, 1, h, w)
        x = x.float().div_(255.).repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
        with torch.amp.autocast("cuda"):
            if kind == "resnet":
                f = backbone(x).float().mean(dim=(2, 3))
            else:
                f = backbone(pixel_values=x).last_hidden_state[:, 0]
            o = torch.sigmoid(head(f.reshape(b, s, -1)))
        preds[at:at + len(o)] = o.float().cpu().numpy()
        at += len(o)
    return preds


results = {k: {} for k in PER_PLANE}
timing = {}
for tag, path in CKPTS.items():
    backbone, head, kind = load(path)
    for k in PER_PLANE:
        keep = keep_indices(k)
        t = time.time()
        p = predict(backbone, head, kind, keep)
        results[k][tag] = p
        timing.setdefault(k, {})[tag] = time.time() - t
        print(f"  {tag:<14} {k:>2}/plane ({len(keep):>2} slices)  "
              f"gold {gold_auc(p):.4f}  {time.time()-t:.0f}s", flush=True)
    del backbone, head
    torch.cuda.empty_cache()

print("\n" + "=" * 68)
print(f"{'slices/plane':<14}{'total':>7}{'ensemble gold':>16}{'vs 12':>9}"
      f"{'rel GPU time':>14}")
print("=" * 68)
base_gold = gold_auc(np.mean(list(results[12].values()), axis=0))
base_time = sum(timing[12].values())
for k in PER_PLANE:
    g = gold_auc(np.mean(list(results[k].values()), axis=0))
    t = sum(timing[k].values())
    print(f"{k:<14}{k*3:>7}{g:>16.4f}{g-base_gold:>+9.4f}"
          f"{t/base_time:>13.2f}x", flush=True)

print("\nIf 8/plane costs under ~0.004 gold for ~0.67x the time, the 9 h cap")
print("buys 50% more members and the 0.8802 six-model becomes affordable.")
np.savez_compressed("/kaggle/working/slice_budget.npz",
                    per_plane=np.array(PER_PLANE),
                    names=np.array(list(CKPTS)),
                    preds=np.stack([np.stack([results[k][t] for t in CKPTS])
                                    for k in PER_PLANE]))
print("saved slice_budget.npz\nDONE", flush=True)
