"""v21: the channel experiment redone with conv1 UNFROZEN. GPU, ~55 min.

The first run of this comparison had a flaw. It unfroze only layer3/layer4, and
in `nn.Sequential(*resnet50().children()[:-2])` that leaves **b[0]=conv1 and
b[1]=bn1 frozen**. So alien channels were fed to a first convolution pretrained
on replicated grayscale, with no ability to adapt -- and both alternatives lost:

    2.5D neighbouring slices    -0.029
    eikonal + edge speed        -0.0563

It was not undertraining. Per-epoch, the control-vs-eikonal gap stopped closing
at epoch 3 and sat flat at ~0.055, which is what a frozen input layer looks like,
not slow learning. More epochs could not have helped; an adaptable conv1 might.

So this reruns all three arms with **conv1 and bn1 trainable** (~9.4k extra
parameters, negligible against the 22.1M already unfrozen):

    control   [I, I, I]
    25d       [prev, I, next]
    eikonal   [I, arrival-time, edge-speed]

Everything else identical to the first run -- sagittal 12 slices, RadImageNet,
layer3+4, same teacher, same rng(0) split, same seed -- so the comparison is
against the earlier numbers as well as within itself.

**What each outcome would mean.** If the gaps shrink toward zero, the first run
measured a frozen input layer rather than the channels. If they stay near -0.03
and -0.056, the channels genuinely carry less than a replicated slice and the
idea is finished for good.
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
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

assert torch.cuda.is_available(), "no CUDA after install"
dev = "cuda"
EPOCHS = 8
BATCH_STUDIES = 4          # 12 slices/study, so a larger batch fits
LR_HEAD, LR_BACKBONE = 3e-4, 2e-5
SAG = 12                   # sagittal block is the first 12 of the 36
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True)
        if "slices" in q]
_eik = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True)
        if "eikonal" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True)
        if "index" in q and "eikonal" not in q]
_eidx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True)
         if "eikonal" in q]
assert _npy and _eik and _idx and _eidx, f"missing inputs: {_npy} {_eik}"
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
LLM = glob.glob("/kaggle/input/**/report_labels_v2.csv", recursive=True)[0]
FALLBACK = glob.glob("/kaggle/input/**/labels_rules_v5.csv", recursive=True)[0]
W = glob.glob("/kaggle/input/**/ResNet50.pt", recursive=True)[0]

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
E = np.load(_eik[0], mmap_mode="r")
emeta = np.load(_eidx[0], allow_pickle=True)
print(f"slices {X.shape} | eikonal {E.shape}", flush=True)
assert X.shape[-1] == 224, f"expected 224px slices, got {X.shape[-1]}"
assert E.shape[0] == X.shape[0] == len(ids), "study counts differ"
assert E.shape[1] == SAG, f"expected {SAG} eikonal slices, got {E.shape[1]}"
# the eikonal cache must describe the same studies in the same order, or every
# channel would be paired with a different knee
assert np.array_equal(np.asarray(emeta["study_ids"]), np.asarray(ids)), \
    "eikonal study order differs from the slice cache"
print("study order matches between caches", flush=True)

lab = pd.read_csv(LLM).set_index("StudyInstanceUID")[LABELS]
_fb = pd.read_csv(FALLBACK).set_index("StudyInstanceUID")[LABELS]
_miss = [u for u in ids if u not in lab.index]
if _miss:
    lab = pd.concat([lab, _fb.loc[_miss]])
G = lab.loc[ids].values.astype(np.float32)

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
train = np.setdiff1d(np.arange(len(ids)), val)
gm = gold_mask[val]
print(f"train {len(train)} | val {len(val)} incl. {gm.sum()} gold", flush=True)


class Studies(Dataset):
    def __init__(self, index):
        self.index = index

    def __getitem__(self, k):
        i = self.index[k]
        return (torch.from_numpy(np.array(X[i][:SAG])),
                torch.from_numpy(np.array(E[i])),
                torch.from_numpy(G[i]))

    def __len__(self):
        return len(self.index)


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
        a = torch.einsum("bsd,cd->bcs", e, self.query) * e.shape[-1] ** -0.5
        pooled = torch.einsum("bcs,bsh->bch", a.softmax(-1), self.value(e))
        return (pooled * self.out).sum(-1) + self.bias


RAD = torch.load(W, map_location="cpu", weights_only=False)
RAD = {k[len("backbone."):]: v for k, v in RAD.items() if k.startswith("backbone.")}


def build():
    net = torchvision.models.resnet50(weights=None)
    b = nn.Sequential(*list(net.children())[:-2])
    b.load_state_dict(RAD, strict=True)
    for p in b.parameters():
        p.requires_grad = False
    # conv1 (b[0]) and bn1 (b[1]) are unfrozen HERE and were not in the first
    # run. That is the whole point: alien channels need an input layer that can
    # adapt to them. Together they are ~9.4k parameters, so this barely touches
    # the capacity argument.
    for blk in (b[0], b[1], b[6], b[7]):
        for p in blk.parameters():
            p.requires_grad = True
    return b.to(dev), AttnHead(2048, len(LABELS)).to(dev)


def make_input(img, eik, mode):
    """[B,12,224,224] uint8 (+ [B,12,56,56] uint8) -> [B*12, 3, 224, 224]."""
    b, s, h, w = img.shape
    i = img.reshape(b * s, 1, h, w).float().div_(255.)
    if mode == "control":
        x = i.repeat(1, 3, 1, 1)
    elif mode == "25d":
        v = img.reshape(b, s, h, w).float().div_(255.)
        prev = torch.cat([v[:, :1], v[:, :-1]], dim=1)
        nxt = torch.cat([v[:, 1:], v[:, -1:]], dim=1)
        x = torch.stack([prev, v, nxt], dim=2).reshape(b * s, 3, h, w)
    else:
        e = eik.reshape(b * s, 1, eik.shape[-2], eik.shape[-1]).float().div_(255.)
        e = F.interpolate(e, size=(h, w), mode="bilinear", align_corners=False)
        # edge speed: 1 - normalised gradient magnitude, the field the solver
        # would slow down on. Cheap to recompute here from the intensity.
        gy = torch.zeros_like(i); gx = torch.zeros_like(i)
        gy[:, :, 1:-1] = (i[:, :, 2:] - i[:, :, :-2]) * 0.5
        gx[:, :, :, 1:-1] = (i[:, :, :, 2:] - i[:, :, :, :-2]) * 0.5
        g = torch.sqrt(gy * gy + gx * gx)
        g = g / g.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        x = torch.cat([i, e, 1.0 - g], dim=1)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def gold_auc(p):
    out = []
    for j in range(len(LABELS)):
        y = Y_gold[val][gm, j]
        if len(np.unique(y)) == 2:
            out.append(roc_auc_score(y, p[gm, j]))
    return float(np.mean(out))


def run(mode, tag):
    torch.manual_seed(0)
    np.random.seed(0)
    backbone, head = build()
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

    def fwd(img, eik):
        b = img.shape[0]
        x = make_input(img.to(dev, non_blocking=True),
                       eik.to(dev, non_blocking=True), mode)
        f = backbone(x).mean(dim=(2, 3))
        return head(f.reshape(b, SAG, -1))

    best, best_p = -1, None
    t = time.time()
    for ep in range(EPOCHS):
        backbone.train(); head.train()
        for img, eik, tgt in loader:
            with torch.amp.autocast("cuda"):
                loss = lossf(fwd(img, eik), tgt.to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
        backbone.eval(); head.eval()
        preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
        vl = DataLoader(Studies(val), batch_size=BATCH_STUDIES, shuffle=False,
                        num_workers=2)
        at = 0
        with torch.no_grad():
            for img, eik, _ in vl:
                with torch.amp.autocast("cuda"):
                    o = torch.sigmoid(fwd(img, eik))
                preds[at:at + len(o)] = o.float().cpu().numpy()
                at += len(o)
        g = gold_auc(preds)
        if g > best:
            best, best_p = g, preds
            torch.save({"backbone": backbone.state_dict(),
                        "head": head.state_dict(), "labels": LABELS,
                        "size": 224, "slices": SAG, "planes": ["Sagittal"],
                        "arch": "resnet50_radimagenet", "variant": tag,
                        "input_mode": mode, "gold": g},
                       f"/kaggle/working/finetuned_{tag}.pt")
        print(f"    {tag} epoch {ep+1}/{EPOCHS} gold {g:.4f} "
              f"{(time.time()-t)/60:.0f}m", flush=True)
    del backbone, head
    torch.cuda.empty_cache()
    return best, best_p


print("\n" + "=" * 62, flush=True)
res = {}
for mode, tag in [("control", "chan_control"), ("25d", "chan_25d"),
                  ("eikonal", "chan_eik")]:
    g, p = run(mode, tag)
    res[mode] = (g, p)
    print(f"  {tag:<14} best gold {g:.4f}", flush=True)

print("=" * 62)
c = res["control"][0]
print(f"  control [I,I,I]              {c:.4f}")
for m, label in [("25d", "2.5D neighbours"), ("eikonal", "eikonal+edge")]:
    print(f"  {label:<28} {res[m][0]:.4f}   {res[m][0]-c:+.4f}")
print()
print("  With conv1 FROZEN the same comparison gave 2.5D -0.029 and")
print("  eikonal -0.0563. If those gaps shrink here, the earlier runs were")
print("  measuring a frozen input layer rather than the channels themselves.")

np.savez_compressed("/kaggle/working/val_predictions_chan.npz",
                    val_index=val,
                    names=np.array(["v21_ctl", "v21_25d", "v21_eik"]),
                    plain=np.stack([res["control"][1], res["25d"][1],
                                    res["eikonal"][1]]))
print("saved val_predictions_chan.npz\nDONE", flush=True)
