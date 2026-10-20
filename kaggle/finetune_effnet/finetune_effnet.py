"""v15: EfficientNet-B3 -- a third architecture family for the ensemble. GPU.

The ensemble rule this project settled on is **pick members by correlation, not
accuracy**:

    v13 336px       gold 0.8533   corr 0.934   added +0.0004 (nothing)
    v11 base        gold 0.8541   corr 0.934   added +0.010 LB
    v14 RadImageNet gold 0.8508   corr 0.856   added +0.016 LB  <- lowest gold

RadImageNet has the *worst* individual gold of the three and gave the biggest
gain, because a CNN pretrained on radiology fails differently from a ViT
pretrained on photographs. Being different beats being good.

So this adds a third family. EfficientNet-B3 is compound-scaled with
depthwise-separable convolutions -- structurally unlike both ResNet-50's
bottleneck stacks and DINOv2's attention -- and is what the top public notebooks
reach for after RadImageNet.

**Honest caveat: this is ImageNet-pretrained, not RadImageNet.** Only the
architecture axis differs from v14, not the pretraining-domain axis, so expect
less decorrelation than RadImageNet bought. The medical option was
`sjudal/radimagenet-densenet121`, which would have differed on both axes, but it
ships as a Keras .h5 -- converting DenseNet to PyTorch means transposing conv
weights, reordering BN parameters and reproducing the dense-concat layout
exactly, and a silently wrong conversion yields a plausible-looking broken
backbone. Not worth that risk while a clean option exists. Revisit if a PyTorch
port appears.

Held at v10/v14: LLM labels (pilkwang), same attention head, same rng(0) split,
same seed, same 224px cache. B3 natively prefers 300px; 224 is used so the
existing cache serves it and so this member stays comparable to the others.

BATCH_STUDIES = 2, as for v14: depthwise convolutions are memory-hungry and an
OOM two hours in costs more than a smaller batch.

**v1 of this kernel died with `ValueError: Input contains NaN` after epoch 1.**
EfficientNet is a well-known fp16-instability case -- SiLU and squeeze-excite
blocks overflow under AMP far more readily than ResNet bottlenecks or ViT
attention -- and the P100 is sm_60, so bfloat16 is not available as the usual
escape. Three fixes, in order of importance:

  1. **Pool in fp32.** `backbone(x).mean(dim=(2,3))` sums 1536x7x7 fp16
     activations per image; that reduction is the most likely overflow site.
  2. **Skip non-finite losses and clip gradients to 1.0.** GradScaler already
     skips inf/nan *gradients*, but it does nothing about a forward pass that
     returns NaN, which then poisons the weights permanently.
  3. **evaluate() reports non-finite predictions instead of raising.** The first
     attempt crashed inside roc_auc_score and threw away a whole epoch of
     diagnostic information -- the log never even printed the training loss, so
     there was no way to tell divergence from overflow. A metric function should
     never be the thing that kills a two-hour run.

**v2 completed but was not good enough to use.** gold 0.7760 against v14's
0.8508, and it still emitted 0.2-0.4% non-finite predictions every epoch -- the
fp16 instability was contained, never cured, so it trained through corrupted
batches the whole way.

The ensemble search made the diagnosis precise. v2's correlation with the shipped
average was **0.821**, the most decorrelated usable member we have (RadImageNet
is 0.916). Its problem is quality, not diversity:

    best combination containing v15_effnet   0.8724
    shipped 4-model                          0.8756   (LB 0.896)

That also sharpened the ensemble rule: **decorrelation is necessary but not
sufficient.** v9_small is even more decorrelated (0.768) and also hurts (-0.0026).
Both weak members sit ~0.07 below the best member, while base and RadImageNet sit
within 0.007. A member must be roughly competitive before its diversity pays.

So v3 targets quality while keeping the diversity:
  * **fp32 throughout** -- no autocast, no GradScaler. ~2x slower but there is no
    bf16 on sm_60 and contained-but-present NaN is not good enough.
  * **unfreeze features[5:]** instead of [6:], and LR_BACKBONE 2e-5 -> 5e-5. A
    CNN trunk tolerates more adaptation than a ViT, and v2's head was working
    through a barely-adapted trunk.

If this reaches ~0.84 at a similar correlation it should be the strongest
addition since RadImageNet. If it stays near 0.78, EfficientNet is done and the
next member should be 5-fold RadImageNet instead.

The real test is not the gold printed at the end -- it is the correlation with
the existing ensemble, measured afterwards by valpreds + the local search.
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
UNFREEZE_FROM = 5          # features[5:] -- one more MBConv stage than v2
LR_HEAD, LR_BACKBONE = 3e-4, 5e-5   # CNN trunk tolerates more than a ViT
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


net = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")
backbone = net.features                      # -> [B, 1536, H/32, W/32]
DIM = 1536
for p in backbone.parameters():
    p.requires_grad = False
for blk in backbone[UNFREEZE_FROM:]:
    for p in blk.parameters():
        p.requires_grad = True
head = AttnHead(DIM, len(LABELS))
backbone, head = backbone.to(dev), head.to(dev)
trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"efficientnet_b3: features[{UNFREEZE_FROM}:] trainable "
      f"({trainable/1e6:.1f}M of {sum(p.numel() for p in backbone.parameters())/1e6:.1f}M)",
      flush=True)


def forward_studies(imgs):
    b, s, h, w = imgs.shape
    x = imgs.reshape(b * s, 1, h, w).float().div_(255.).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    f = backbone(x).float().mean(dim=(2, 3))
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
lossf = nn.BCEWithLogitsLoss()


@torch.no_grad()
def evaluate():
    backbone.eval(); head.eval()
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val), batch_size=BATCH_STUDIES, shuffle=False,
                    num_workers=2)
    at = 0
    for imgs, _ in vl:
        out = forward_studies(imgs.to(dev, non_blocking=True))
        preds[at:at + len(out)] = out.float().cpu().numpy()
        at += len(out)
    n_bad = int((~np.isfinite(preds)).sum())
    if n_bad:
        print(f"    WARNING {n_bad}/{preds.size} non-finite predictions",
              flush=True)
        preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
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
    tot = k = n_skip = 0
    for imgs, tgt in loader:
        imgs, tgt = imgs.to(dev, non_blocking=True), tgt.to(dev)
        loss = lossf(forward_studies(imgs), tgt)
        if not torch.isfinite(loss):
            n_skip += 1
            sched.step()
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in backbone.parameters() if p.requires_grad] +
            list(head.parameters()), 1.0)
        opt.step(); sched.step()
        tot += loss.item(); k += 1
    d, g = evaluate()
    print(f"{ep+1:>5}{tot/max(k,1):>8.4f}{d:>10.4f}{g:>8.4f}"
          f"{(time.time()-t0)/60:>9.1f}m"
          f"{('  skipped ' + str(n_skip)) if n_skip else ''}", flush=True)
    if g > best:
        best = g
        torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                    "labels": LABELS, "size": int(X.shape[-1]),
                    "slices": int(n_slices // 3), "arch": "efficientnet_b3",
                    "dim": DIM, "planes": ["Sagittal", "Coronal", "Axial"],
                    "gold": g},
                   "/kaggle/working/finetuned_effnet.pt")

print(f"\nbest gold {best:.4f}  (v10 small 0.8438, v14 radimagenet 0.8508)",
      flush=True)
print("what matters is its correlation with the ensemble, measured next.",
      flush=True)
print("saved /kaggle/working/finetuned_effnet.pt\nDONE", flush=True)
