"""v19B: the same recipe on 10% more data. GPU, ~60 min.

The ensemble has plateaued -- all fifteen members now land within +/-0.001 of the
shipped combination, so more members of the same kind are worthless. This is not
a new member; it is the existing one trained on data we were throwing away.

**Every model in this project has held out 451 studies. Only 58 of them are the
gold ruler.** The other 393 were excluded for no reason beyond being in the same
array -- `evaluate()` computes a derived-AUC over them, but that column is a
diagnostic, never a selection signal, and checkpointing has always been on gold.

So this holds out gold-58 alone and trains on 4349 studies instead of 3956:
**+10% data, zero extra inference cost.**

gold-58 stays out. Training on it would destroy the only ruler this project has,
and every measurement made since would stop meaning anything.

Expected gain is modest and worth stating so the result is judged fairly: the
1500 -> 4407 jump was +194% data for +0.031 gold, so +10% predicts roughly
+0.003. That is inside gold's noise for a single model, which is why the honest
test is the *ensemble* margin afterwards, not this model's own number.

Variant B recipe unchanged from v17B so the comparison is single-variable.
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
LR_HEAD, LR_BACKBONE = 3e-4, 2e-5
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
# VARIANT is set per run below; both teachers are attached
PILK = glob.glob("/kaggle/input/**/report_labels_v2.csv", recursive=True)[0]
LEEH = glob.glob("/kaggle/input/**/llm_labels_v4_blend.csv", recursive=True)[0]
VARIANT = "B"   # hardcoded: Kaggle kernels take no env vars
LLM = LEEH if VARIANT == "A" else PILK
UNFREEZE_L3 = (VARIANT == "B")
print(f"VARIANT {VARIANT}: teacher={'leehans' if VARIANT=='A' else 'pilkwang'}, "
      f"unfreeze={'layer3+4' if UNFREEZE_L3 else 'layer4'}", flush=True)
FALLBACK = glob.glob("/kaggle/input/**/labels_rules_v5.csv", recursive=True)[0]
W = glob.glob("/kaggle/input/**/ResNet50.pt", recursive=True)[0]
print(f"weights: {W}", flush=True)

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
n_slices = X.shape[1]
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
print(f"gold studies: {gold_mask.sum()}", flush=True)


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


# resnet50() with no weights needs no internet; RadImageNet supplies them
net = torchvision.models.resnet50(weights=None)
backbone = nn.Sequential(*list(net.children())[:-2])       # -> [B, 2048, 7, 7]
sd = torch.load(W, map_location="cpu", weights_only=False)
sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
backbone.load_state_dict(sd, strict=True)                  # strict: no silent misses
print(f"loaded {len(sd)} RadImageNet tensors", flush=True)

for p in backbone.parameters():
    p.requires_grad = False
for p in backbone[7].parameters():                         # layer4
    p.requires_grad = True
if UNFREEZE_L3:
    for p in backbone[6].parameters():                     # layer3
        p.requires_grad = True
head = AttnHead(2048, len(LABELS))
backbone, head = backbone.to(dev), head.to(dev)
trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"backbone: layer4 trainable ({trainable/1e6:.1f}M params)", flush=True)


def forward_studies(imgs):
    b, s, h, w = imgs.shape
    x = imgs.reshape(b * s, 1, h, w).float().div_(255.).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    f = backbone(x).mean(dim=(2, 3))                       # global average pool
    return head(f.reshape(b, s, -1))


rng = np.random.default_rng(0)
order = rng.permutation(len(ids))
# FULL-DATA: hold out ONLY the 58 gold studies, not the extra 393.
# `val` has always been 58 gold + 393 non-gold, and those 393 were never used
# for anything except being excluded -- the derived-AUC column is a diagnostic,
# not a selection signal. Releasing them is +10% training data at zero runtime
# cost. gold-58 stays held out because it is the only ruler that exists; train
# on it and every number this project relies on becomes meaningless.
val = np.where(gold_mask)[0]
train = np.setdiff1d(np.arange(len(ids)), val)
assert not np.intersect1d(train, np.where(gold_mask)[0]).size, \
    "gold studies leaked into training -- the ruler would be worthless"
print(f"train {len(train)} | val {len(val)} incl. {gold_mask.sum()} gold", flush=True)

loader = DataLoader(Studies(train), batch_size=BATCH_STUDIES, shuffle=True,
                    num_workers=2, drop_last=True)
opt = torch.optim.AdamW([
    {"params": [p for p in backbone.parameters() if p.requires_grad],
     "lr": LR_BACKBONE},
    {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=EPOCHS * len(loader))
scaler = torch.amp.GradScaler("cuda")
lossf = nn.BCEWithLogitsLoss()


@torch.no_grad()
def evaluate():
    backbone.eval(); head.eval()
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val), batch_size=BATCH_STUDIES, shuffle=False,
                    num_workers=2)
    at = 0
    for imgs, _ in vl:
        with torch.amp.autocast("cuda"):
            out = forward_studies(imgs.to(dev, non_blocking=True))
        preds[at:at + len(out)] = out.float().cpu().numpy()
        at += len(out)
    gm = gold_mask[val]
    gold, derived = [], []
    for j in range(len(LABELS)):
        yb = (G[val][:, j] >= 0.5).astype(int)
        if len(np.unique(yb)) == 2:
            derived.append(roc_auc_score(yb, preds[:, j]))
        yg = Y_gold[val][gm, j]
        if len(np.unique(yg)) == 2:
            gold.append(roc_auc_score(yg, preds[gm, j]))
    backbone.train(); head.train()
    return float(np.mean(derived)), float(np.mean(gold))


print("\nepoch   loss   derived   gold      time", flush=True)
best = -1
t0 = time.time()
for ep in range(EPOCHS):
    backbone.train(); head.train()
    tot = k = 0
    for imgs, tgt in loader:
        imgs, tgt = imgs.to(dev, non_blocking=True), tgt.to(dev)
        with torch.amp.autocast("cuda"):
            loss = lossf(forward_studies(imgs), tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item(); k += 1
    d, g = evaluate()
    print(f"{ep+1:>5}{tot/max(k,1):>8.4f}{d:>10.4f}{g:>8.4f}"
          f"{(time.time()-t0)/60:>9.1f}m", flush=True)
    if g > best:
        best = g
        torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                    "labels": LABELS, "size": int(X.shape[-1]),
                    "slices": int(n_slices // 3), "arch": "resnet50_radimagenet", "variant": "fullB",
                    "planes": ["Sagittal", "Coronal", "Axial"], "gold": g},
                   f"/kaggle/working/finetuned_full_B.pt")

print(f"\nbest gold {best:.4f}  (v10 DINOv2-small on the same labels: 0.8438)",
      flush=True)
print("saved /kaggle/working/finetuned_radimagenet.pt\nDONE", flush=True)
