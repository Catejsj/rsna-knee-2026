"""v14: RadImageNet ResNet-50 -- a backbone that is not DINOv2 at all. GPU.

Ensembling has one measured lever on this project, and it is **encoder
diversity**:

    v10 small + v11 base + v12 leehans   gold 0.8648  ->  LB 0.880
    the same three plus v13 336px        gold 0.8653  (+0.0004, nothing)

336px was individually strong (gold 0.8533) and still added nothing, because it
is *still DINOv2-small* -- higher resolution, same errors. Meanwhile base, which
is individually WORSE than small on the leaderboard (0.863 vs 0.870), lifted the
ensemble by +0.010. Being different matters more than being good.

So the next member has to differ in kind. RadImageNet is a ResNet-50 pretrained
on ~1.35M radiology images (CT/MR/US) rather than on natural photographs. A CNN
trained on medical scans should fail differently from a ViT trained on ImageNet
photos, which is exactly what the average needs. Every top public notebook pairs
DINO with this.

Held constant against v10 so the comparison stays clean: LLM labels (pilkwang),
the same per-diagnosis attention head, the same rng(0) split, the same seed, the
same 224px x 36-slice cache.

Two deliberate differences, both forced:

  * **BATCH_STUDIES = 2, not 4.** ResNet-50 activations at 144 images per step
    are far heavier than DINOv2-small's, and an OOM two hours in costs more than
    a smaller batch does. The learning rates are unchanged, so this is a real
    difference from the DINOv2 runs -- it is not a controlled comparison of
    architectures, and is not claimed as one. The goal is a decorrelated member,
    not a like-for-like benchmark.
  * **Unfreeze layer4 only.** That is ~15M trainable parameters against the
    DINOv2 runs' 7.1M. Unfreezing layer3 as well would triple it, and "add
    information, not parameters" has survived every test on this project
    including v11-base.

Weights are `marwanmath/resnet-50-radimagenet-marwan`, an OrderedDict of
`backbone.N.*` matching nn.Sequential(*resnet50().children()[:-2]): conv1, bn1,
then layer1-4 with 60/78/114/60 tensors, i.e. ResNet-50's 3/4/6/3 bottlenecks.
Verified locally before this was written; the load below is strict so a silent
partial match cannot happen.
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
LLM = glob.glob("/kaggle/input/**/report_labels_v2.csv", recursive=True)[0]
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
val = np.unique(np.concatenate([np.where(gold_mask)[0],
                                order[:400][~gold_mask[order[:400]]]]))
train = np.setdiff1d(np.arange(len(ids)), val)
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
                    "slices": int(n_slices // 3), "arch": "resnet50_radimagenet",
                    "planes": ["Sagittal", "Coronal", "Axial"], "gold": g},
                   "/kaggle/working/finetuned_radimagenet.pt")

print(f"\nbest gold {best:.4f}  (v10 DINOv2-small on the same labels: 0.8438)",
      flush=True)
print("saved /kaggle/working/finetuned_radimagenet.pt\nDONE", flush=True)
