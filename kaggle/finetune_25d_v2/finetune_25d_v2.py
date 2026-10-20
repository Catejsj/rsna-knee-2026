"""v23: 2.5D with slices that are actually adjacent. GPU, ~1.5 h (2 arms).

The third and final attempt at 2.5D, and the first one where the premise is
measured rather than assumed. Two earlier runs put "neighbouring" slices in the
RGB channels and lost (-0.029, then -0.0142 once conv1 was unfrozen). Reading the
cache code afterwards showed those neighbours were **~7-8 mm apart** -- the
loader subsamples a ~40-file series to 32, then the stacker takes 12 anchors
across the middle 70%. `rsna-cache-25d` prints both gap distributions; check them
in its log before reading anything below.

### The control comes out of the same array

`cache_25d` stores `[n, 12 anchors * 3 offsets, 224, 224]` with offsets
prev/anchor/next, so:

    ctrl      channels = (anchor, anchor, anchor)   -- offset 1, replicated
    true25d   channels = (prev, anchor, next)       -- offsets 0, 1, 2

Same studies, same anchors, same crop, same laterality flip, same file. The only
difference between the two numbers is whether the outer channels carry the
physical neighbours. The earlier attempts compared across two separately built
caches, which left room for exactly this kind of confound.

### conv1 is trainable in BOTH arms

Non-negotiable, and the reason the previous measurement was wrong by half. With
conv1 frozen you are measuring a first layer that was trained on RGB photographs
being handed something that is not RGB. It is only ~9.4k parameters. The control
gets it too, so it is not a free variable.

### Sagittal only

That is what the cache holds (three planes of triplets would be 24 GB, past the
output cap). So the absolute gold here will sit BELOW the 3-plane models -- one
plane, 12 slices instead of 36. **Do not compare these numbers to v19B's 0.8700.**
The only comparison that means anything is ctrl vs true25d, and both are
sagittal-only.

If true25d wins, the follow-up is a 3-plane cache and a real ensemble member. If
it loses again with genuine neighbours and a trainable conv1, 2.5D is closed for
good and there is nothing left to blame.
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
SEED = 0
ANCHORS, OFFSETS = 12, 3
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True)
        if "slices25d" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True)
        if "slices25d_index" in q]
assert _npy and _idx, "the 2.5D cache is not attached (looking for slices25d_*)"
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
PILK = glob.glob("/kaggle/input/**/report_labels_v2.csv", recursive=True)[0]
FALLBACK = glob.glob("/kaggle/input/**/labels_rules_v5.csv", recursive=True)[0]
W = glob.glob("/kaggle/input/**/ResNet50.pt", recursive=True)[0]

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
present = meta["present"]
X = np.load(_npy[0], mmap_mode="r")
print(f"cache {X.shape} | {len(ids)} studies | layout {meta['layout']}", flush=True)
assert X.shape[1] == ANCHORS * OFFSETS, \
    f"expected {ANCHORS*OFFSETS} slices/study, got {X.shape[1]}"
assert X.shape[-1] == 224, f"expected 224px, got {X.shape[-1]}px"

# The premise, restated here so it appears in THIS log too rather than only in
# the cache's. If these are close, the result below is uninformative.
tg = meta["true_gap_mm"][np.isfinite(meta["true_gap_mm"])]
ag = meta["anchor_gap_mm"][np.isfinite(meta["anchor_gap_mm"])]
if len(tg) and len(ag):
    print(f"gap: true neighbours {np.median(tg):.2f} mm vs anchors "
          f"{np.median(ag):.2f} mm ({np.median(ag)/max(np.median(tg),1e-6):.1f}x)",
          flush=True)

lab = pd.read_csv(PILK).set_index("StudyInstanceUID")[LABELS]
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
print(f"gold studies: {gold_mask.sum()} | with sagittal: "
      f"{(gold_mask & present).sum()}", flush=True)


class Studies(Dataset):
    """Returns [12, 3, 224, 224] -- one RGB triplet per anchor.

    `mode` picks the channels: 'true25d' takes prev/anchor/next, 'ctrl' repeats
    the anchor three times. Both read the same rows of the same array.
    """

    def __init__(self, index, mode):
        self.index, self.mode = index, mode

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = self.index[k]
        raw = np.asarray(X[i]).reshape(ANCHORS, OFFSETS, X.shape[-2], X.shape[-1])
        if self.mode == "ctrl":
            raw = np.repeat(raw[:, 1:2], OFFSETS, axis=1)
        return torch.from_numpy(np.ascontiguousarray(raw)), torch.from_numpy(G[i])


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


def build():
    net = torchvision.models.resnet50(weights=None)
    bb = nn.Sequential(*list(net.children())[:-2])
    sd = torch.load(W, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    bb.load_state_dict(sd, strict=True)
    for p in bb.parameters():
        p.requires_grad = False
    for p in bb[7].parameters():                    # layer4
        p.requires_grad = True
    for p in bb[6].parameters():                    # layer3
        p.requires_grad = True
    # conv1 + bn1: the whole point. ~9.4k params, given to BOTH arms so it is
    # not a free variable -- with it frozen you measure an RGB-photo filter bank
    # being fed something that is not RGB, which is what went wrong before.
    for p in bb[0].parameters():
        p.requires_grad = True
    for p in bb[1].parameters():
        p.requires_grad = True
    return bb.to(dev), AttnHead(2048, len(LABELS)).to(dev)


def forward_studies(backbone, head, imgs):
    b, s, c, h, w = imgs.shape
    x = imgs.reshape(b * s, c, h, w).float().div_(255.)
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    f = backbone(x).mean(dim=(2, 3))
    return head(f.reshape(b, s, -1))


# Studies with no sagittal series are all-zero rows; training on blank images
# teaches the head that "blank" predicts the label prior. Drop them from BOTH
# arms so the comparison stays matched.
usable = np.where(present)[0]
val = np.array([i for i in usable if gold_mask[i]])
train = np.array([i for i in usable if not gold_mask[i]])
assert not set(val) & set(train), "gold leaked into training"
print(f"train {len(train)} | val {len(val)} gold studies", flush=True)


@torch.no_grad()
def evaluate(backbone, head, mode):
    backbone.eval(); head.eval()
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val, mode), batch_size=BATCH_STUDIES, shuffle=False,
                    num_workers=2)
    at = 0
    for imgs, _ in vl:
        with torch.amp.autocast("cuda"):
            out = forward_studies(backbone, head, imgs.to(dev, non_blocking=True))
        preds[at:at + len(out)] = out.float().cpu().numpy()
        at += len(out)
    gold = []
    for j in range(len(LABELS)):
        yg = Y_gold[val][:, j]
        if len(np.unique(yg)) == 2 and np.isfinite(preds[:, j]).all():
            gold.append(roc_auc_score(yg, preds[:, j]))
    backbone.train(); head.train()
    return (float(np.mean(gold)) if gold else float("nan")), preds


def run_arm(mode):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    backbone, head = build()
    loader = DataLoader(Studies(train, mode), batch_size=BATCH_STUDIES,
                        shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad],
         "lr": LR_BACKBONE},
        {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=EPOCHS * len(loader))
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.BCEWithLogitsLoss()

    print(f"\n=== arm {mode} ===", flush=True)
    print("epoch   loss     gold      time", flush=True)
    best, best_preds = -1.0, None
    t0 = time.time()
    for ep in range(EPOCHS):
        backbone.train(); head.train()
        tot = 0.0
        k = 0
        for imgs, tgt in loader:
            imgs, tgt = imgs.to(dev, non_blocking=True), tgt.to(dev)
            with torch.amp.autocast("cuda"):
                loss = lossf(forward_studies(backbone, head, imgs), tgt)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); k += 1
        g, preds = evaluate(backbone, head, mode)
        print(f"{ep+1:>5}{tot/max(k,1):>8.4f}{g:>9.4f}"
              f"{(time.time()-t0)/60:>9.1f}m", flush=True)
        if g > best:
            best, best_preds = g, preds.copy()
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "labels": LABELS, "size": int(X.shape[-1]),
                        "slices": ANCHORS, "arch": "resnet50_radimagenet",
                        "variant": f"25d_{mode}", "mode": mode,
                        "planes": ["Sagittal"], "gold": g},
                       f"/kaggle/working/finetuned_25d_{mode}.pt")
    print(f"  arm {mode}: best gold {best:.4f} in {(time.time()-t0)/60:.0f}m",
          flush=True)
    return best, best_preds


res, pr = {}, {}
for mode in ("ctrl", "true25d"):
    res[mode], pr[mode] = run_arm(mode)

print("\n" + "=" * 56, flush=True)
print(f"ctrl     (anchor,anchor,anchor)  gold {res['ctrl']:.4f}", flush=True)
print(f"true25d  (prev,anchor,next)      gold {res['true25d']:.4f}", flush=True)
print(f"delta                            {res['true25d']-res['ctrl']:+.4f}", flush=True)
print("\nBoth sagittal-only -- do NOT compare to the 3-plane v19B (0.8700).")
print("Gold deltas under ~0.015 carry no information on 58 studies; a result "
      "inside that band is unresolved, not a win.", flush=True)
if len(pr) == 2:
    c = np.corrcoef(pr["ctrl"].ravel(), pr["true25d"].ravel())[0, 1]
    print(f"arm correlation {c:.3f}", flush=True)

np.savez_compressed("/kaggle/working/preds25d.npz", study_ids=ids[val],
                    **{f"pred_{k}": v for k, v in pr.items()})
print("DONE", flush=True)
