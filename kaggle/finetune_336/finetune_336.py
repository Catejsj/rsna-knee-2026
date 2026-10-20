"""v13: 336px at three planes -- the last untested representation axis. GPU.

Resolution was never tested fairly. The single 336px run used **one plane** and
lost (gold 0.6523 vs 224px x 3 planes 0.6811), but comparing like with like
showed the deficit tracked *planes*, not resolution. 336px x 3 planes has been
open since, and the 0.899 public notebook caches at 336px.

Labels are held at pilkwang (the v10 teacher, LB 0.870) rather than the nominally
better leehans set, because v12 showed the label lever has saturated: a teacher
+0.023 better produced a model only +0.004 better. Holding the teacher constant
against v10 makes this a clean single-variable test of resolution.

Everything else is v10: small encoder, 4 unfrozen blocks, same head, same
schedule, same split, same seed. Compare against v10's gold 0.8438.

Two hazards worth stating:

  * The 336 cache is written as `slices_224_12x3.npy` -- the filename is
    hardcoded upstream and is wrong. Only the array shape and the index npz
    describe it honestly, so this kernel asserts on the shape.
  * `size` in the checkpoint used to be the literal 224. At 336 that would make
    the submission kernel resize test images to 224 while the weights expect
    336, silently and with no error. It now reads from the cache.

Budget: 336px is 2.25x the compute of 224px, so ~3 h against the 9 h GPU limit.
Do not try this with the base encoder -- 4.2 h x 2.25 would not fit.
"""
import os
import subprocess
import sys
import time

# --- torch first, before anything imports the preinstalled build -------------
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

print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
assert torch.cuda.is_available(), "no CUDA after install"
dev = "cuda"
print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

UNFREEZE_LAST = 4
EPOCHS = 8
BATCH_STUDIES = 4          # studies per step; each is 36 slices
LR_HEAD, LR_BACKBONE = 3e-4, 2e-5
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

print("input tree:", flush=True)
for _p in sorted(glob.glob("/kaggle/input/*")):
    print("  ", _p, [os.path.basename(q) for q in sorted(glob.glob(_p + "/*"))[:8]],
          flush=True)

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
assert _npy and _idx, f"cache not found: npy={_npy} idx={_idx}"
CACHE, IDX = _npy[0], _idx[0]
print(f"cache: {CACHE}", flush=True)
print(f"index: {IDX}", flush=True)
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
LLM = glob.glob("/kaggle/input/**/report_labels_v2.csv", recursive=True)[0]
FALLBACK = glob.glob("/kaggle/input/**/labels_rules_v5.csv", recursive=True)[0]

meta = np.load(IDX, allow_pickle=True)
ids = meta["study_ids"]
X = np.load(CACHE, mmap_mode="r")          # [n, 36, 224, 224] uint8
n_slices = X.shape[1]
print(f"cache {X.shape} | {len(ids)} studies", flush=True)
EXPECT = 336
assert X.shape[-1] == EXPECT, (
    f"expected a {EXPECT}px cache, got {X.shape[-1]}px. NB the 336 cache is "
    f"written as slices_224_12x3.npy -- the filename is a lie, only the shape "
    f"and the index npz are trustworthy. Check which dataset is attached.")

lab = pd.read_csv(LLM).set_index("StudyInstanceUID")[LABELS]
_fb = pd.read_csv(FALLBACK).set_index("StudyInstanceUID")[LABELS]
# the public set covers 4406 of 4407 studies; fill the gap from our rules
_missing = [u for u in ids if u not in lab.index]
if _missing:
    print(f"filling {len(_missing)} study(s) from rules_v5", flush=True)
    lab = pd.concat([lab, _fb.loc[_missing]])
G = lab.loc[ids, LABELS].values.astype(np.float32)
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
        return torch.from_numpy(np.asarray(X[i])), torch.from_numpy(G[i]), i


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    return imgs, torch.stack([b[1] for b in batch]), [b[2] for b in batch]


class AttnHead(nn.Module):
    """One learned query per diagnosis, attending over a study's slices."""

    def __init__(self, dim, n_out, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(n_out, dim) * dim ** -0.5)
        self.value = nn.Linear(dim, hidden)
        self.out = nn.Parameter(torch.zeros(n_out, hidden))
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.normal_(self.out, std=hidden ** -0.5)

    def forward(self, emb):                       # emb: [B, S, dim]
        e = self.norm(emb)
        attn = torch.einsum("bsd,cd->bcs", e, self.query) * e.shape[-1] ** -0.5
        attn = attn.softmax(-1)                   # [B, n_out, S]
        pooled = torch.einsum("bcs,bsh->bch", attn, self.value(e))
        return (pooled * self.out).sum(-1) + self.bias


from transformers import AutoModel
root = [os.path.dirname(p) for p in
        glob.glob("/kaggle/input/**/config.json", recursive=True)
        if "dinov2" in p.lower()][0]
backbone = AutoModel.from_pretrained(root)
for p in backbone.parameters():
    p.requires_grad = False
for blk in backbone.encoder.layer[-UNFREEZE_LAST:]:
    for p in blk.parameters():
        p.requires_grad = True
dim = backbone.config.hidden_size
head = AttnHead(dim, len(LABELS))
backbone, head = backbone.to(dev), head.to(dev)
trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"backbone: last {UNFREEZE_LAST} blocks trainable ({trainable/1e6:.1f}M params)",
      flush=True)


def forward_studies(imgs):
    """[B, S, H, W] uint8 -> [B, 12] logits."""
    b, s, h, w = imgs.shape
    x = imgs.reshape(b * s, 1, h, w).float().div_(255.).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    emb = backbone(pixel_values=x).last_hidden_state[:, 0]
    return head(emb.reshape(b, s, -1))


rng = np.random.default_rng(0)
order = rng.permutation(len(ids))
val = np.concatenate([np.where(gold_mask)[0],
                      order[:400][~gold_mask[order[:400]]]])
val = np.unique(val)
train = np.setdiff1d(np.arange(len(ids)), val)
print(f"train {len(train)} | val {len(val)} (incl. all {gold_mask.sum()} gold)",
      flush=True)

loader = DataLoader(Studies(train), batch_size=BATCH_STUDIES, shuffle=True,
                    num_workers=2, collate_fn=collate, drop_last=True)
opt = torch.optim.AdamW([
    {"params": [p for p in backbone.parameters() if p.requires_grad],
     "lr": LR_BACKBONE},
    {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=EPOCHS * len(loader))
scaler = torch.amp.GradScaler("cuda")
lossf = nn.BCEWithLogitsLoss()

from sklearn.metrics import roc_auc_score


@torch.no_grad()
def evaluate():
    backbone.eval(); head.eval()
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val), batch_size=BATCH_STUDIES, shuffle=False,
                    num_workers=2, collate_fn=collate)
    at = 0
    for imgs, _, _ in vl:
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
    for imgs, tgt, _ in loader:
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
        torch.save({"backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "labels": LABELS, "size": int(X.shape[-1]),
                    "slices": int(n_slices // 3),
                    "planes": ["Sagittal", "Coronal", "Axial"],
                    "unfreeze_last": UNFREEZE_LAST, "gold": g, "derived": d},
                   "/kaggle/working/finetuned_336.pt")

print(f"\nbest gold {best:.4f}  (frozen encoder + ridge was 0.6811, LB 0.742)",
      flush=True)
print("saved /kaggle/working/finetuned.pt\nDONE", flush=True)
