"""v16: five RadImageNet folds -- decorrelation by data split. GPU, ~3 h.

The ensemble is at a plateau: with the current members no combination beats the
shipped 4-model (gold 0.8756, LB 0.896) by more than +0.0014, which is noise.
Progress needs new members, and today established exactly what qualifies:

    a member must be COMPETITIVE  (within ~0.01 of the best member)
    and DECORRELATED              (corr < ~0.93 with the ensemble average)

Both conditions are load-bearing. Members that met only one failed:

    v13 336px    gold 0.8533  corr 0.934  competitive, not decorrelated  -> +0.0004
    v15 effnet   gold 0.7760  corr 0.821  decorrelated, not competitive  -> -0.0032
    v9  small    gold 0.7811  corr 0.768  decorrelated, not competitive  -> -0.0026
    v14 radimg   gold 0.8508  corr 0.916  BOTH                           -> +0.016 LB

RadImageNet is the one family already proven on both counts. Training it on five
different subsets of the data produces five members that differ in what they have
seen rather than in what they are -- known-good quality, decorrelated from each
other by construction. Lowest-risk way to add several qualifying members at once.

**`val` is held out of every fold.** The folds partition only the remaining pool,
so all five models' predictions on `val` are clean and directly comparable with
the existing members', and the local ensemble search can mix them freely. Each
fold therefore trains on ~3165 studies instead of 3956 -- slightly less data per
model, traded for five of them.

Saves per-fold val predictions directly, so no separate valpreds kernel is needed.
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
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)

EPOCHS = 8
BATCH_STUDIES = 2
FOLDS = 5
LR_HEAD, LR_BACKBONE = 3e-4, 2e-5
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
LLM = glob.glob("/kaggle/input/**/report_labels_v2.csv", recursive=True)[0]
FALLBACK = glob.glob("/kaggle/input/**/labels_rules_v5.csv", recursive=True)[0]
W = glob.glob("/kaggle/input/**/ResNet50.pt", recursive=True)[0]

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
print(f"cache {X.shape} | {len(ids)} studies", flush=True)
assert X.shape[-1] == 224, (
    f"expected the 224px cache, got {X.shape[-1]}px. Both caches are written as "
    f"slices_224_12x3.npy, so only the shape is trustworthy.")

lab = pd.read_csv(LLM).set_index("StudyInstanceUID")[LABELS]
_fb = pd.read_csv(FALLBACK).set_index("StudyInstanceUID")[LABELS]
_missing = [u for u in ids if u not in lab.index]
if _missing:
    print(f"filling {len(_missing)} study(s) from rules_v5", flush=True)
    lab = pd.concat([lab, _fb.loc[_missing]])
G = lab.loc[ids].values.astype(np.float32)

train_csv = pd.read_csv(os.path.join(COMP, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)
gold_mask = np.array([u in gold_ids for u in ids])
Y_gold = np.zeros((len(ids), len(LABELS)), dtype=np.int64)
Y_gold[gold_mask] = gold_df.loc[ids[gold_mask], LABELS].values.astype(int)

# identical to every other kernel -- do not change, comparability depends on it
rng = np.random.default_rng(0)
order = rng.permutation(len(ids))
val = np.unique(np.concatenate([np.where(gold_mask)[0],
                                order[:400][~gold_mask[order[:400]]]]))
pool = np.setdiff1d(np.arange(len(ids)), val)
fold_rng = np.random.default_rng(1234)
perm = fold_rng.permutation(len(pool))
held = [pool[perm[i::FOLDS]] for i in range(FOLDS)]
print(f"pool {len(pool)} | val {len(val)} incl. {gold_mask.sum()} gold | "
      f"{FOLDS} folds, each training on ~{len(pool) - len(held[0])}", flush=True)


class Studies(Dataset):
    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = self.index[k]
        return torch.from_numpy(np.asarray(X[i])), torch.from_numpy(G[i])


class AttnHead(nn.Module):
    def __init__(self, dim, n_out, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(n_out, dim) * dim ** -0.5)
        self.value = nn.Linear(dim, hidden)
        self.out = nn.Parameter(torch.zeros(n_out, hidden))
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.normal_(self.out, std=hidden ** -0.5)

    def forward(self, emb):
        e = self.norm(emb)
        attn = torch.einsum("bsd,cd->bcs", e, self.query) * e.shape[-1] ** -0.5
        pooled = torch.einsum("bcs,bsh->bch", attn.softmax(-1), self.value(e))
        return (pooled * self.out).sum(-1) + self.bias


RAD = torch.load(W, map_location="cpu", weights_only=False)
RAD = {k[len("backbone."):]: v for k, v in RAD.items() if k.startswith("backbone.")}
print(f"RadImageNet tensors: {len(RAD)}", flush=True)


def gold_auc(preds):
    gm = gold_mask[val]
    out = []
    for j in range(len(LABELS)):
        y = Y_gold[val][gm, j]
        if len(np.unique(y)) == 2:
            out.append(roc_auc_score(y, preds[gm, j]))
    return float(np.mean(out))


def build():
    net = torchvision.models.resnet50(weights=None)
    b = nn.Sequential(*list(net.children())[:-2])
    b.load_state_dict(RAD, strict=True)
    for p in b.parameters():
        p.requires_grad = False
    for p in b[7].parameters():
        p.requires_grad = True
    return b.to(dev), AttnHead(2048, len(LABELS)).to(dev)


def fwd(backbone, head, imgs):
    b, s, h, w = imgs.shape
    x = imgs.reshape(b * s, 1, h, w).float().div_(255.).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    f = backbone(x).mean(dim=(2, 3))
    return head(f.reshape(b, s, -1))


@torch.no_grad()
def predict(backbone, head):
    backbone.eval(); head.eval()
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val), batch_size=BATCH_STUDIES, shuffle=False,
                    num_workers=2)
    at = 0
    for imgs, _ in vl:
        with torch.amp.autocast("cuda"):
            o = torch.sigmoid(fwd(backbone, head, imgs.to(dev, non_blocking=True)))
        preds[at:at + len(o)] = o.float().cpu().numpy()
        at += len(o)
    backbone.train(); head.train()
    return preds


def run_fold(f):
    torch.manual_seed(100 + f)
    np.random.seed(100 + f)
    tr = np.setdiff1d(pool, held[f])
    backbone, head = build()
    loader = DataLoader(Studies(tr), batch_size=BATCH_STUDIES, shuffle=True,
                        num_workers=2, drop_last=True)
    opt = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad],
         "lr": LR_BACKBONE},
        {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=EPOCHS * len(loader))
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.BCEWithLogitsLoss()

    best, best_preds = -1, None
    t = time.time()
    for ep in range(EPOCHS):
        for imgs, tgt in loader:
            imgs, tgt = imgs.to(dev, non_blocking=True), tgt.to(dev)
            with torch.amp.autocast("cuda"):
                loss = lossf(fwd(backbone, head, imgs), tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
        p = predict(backbone, head)
        g = gold_auc(p)
        if g > best:
            best, best_preds = g, p
            torch.save({"backbone": backbone.state_dict(),
                        "head": head.state_dict(), "labels": LABELS,
                        "size": int(X.shape[-1]), "slices": int(X.shape[1] // 3),
                        "arch": "resnet50_radimagenet", "fold": f, "gold": g,
                        "planes": ["Sagittal", "Coronal", "Axial"]},
                       f"/kaggle/working/finetuned_radimg_f{f}.pt")
        print(f"    fold {f} epoch {ep+1}/{EPOCHS} gold {g:.4f} "
              f"{(time.time()-t)/60:.0f}m", flush=True)
    del backbone, head
    torch.cuda.empty_cache()
    return best, best_preds


print("\n" + "=" * 60, flush=True)
names, preds = [], []
for f in range(FOLDS):
    b, p = run_fold(f)
    names.append(f"v16_radf{f}")
    preds.append(p)
    print(f"  fold {f}: best gold {b:.4f}", flush=True)

P = np.stack(preds)
print("\n" + "=" * 60)
print(f"mean of the {FOLDS} folds: gold {gold_auc(P.mean(0)):.4f} "
      f"(single RadImageNet was 0.8508)", flush=True)
c = np.corrcoef(np.stack([p[gold_mask[val]].ravel() for p in preds]))
print("fold-to-fold correlation (want well under 1.0, or the split is not "
      "buying diversity):", flush=True)
print(np.round(c, 3), flush=True)

np.savez_compressed("/kaggle/working/val_predictions_radfolds.npz",
                    val_index=val, names=np.array(names), plain=P)
print("\nsaved val_predictions_radfolds.npz + 5 checkpoints\nDONE", flush=True)
