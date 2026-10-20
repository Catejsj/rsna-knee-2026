"""v24: can the image model correct the report labels? AUM. GPU, ~3 h (3 arms).

The ceiling on this project is the teacher. Gold-58 is image-derived, the labels
come from reports, and the reports leave things out -- ~56% of Synovitis
positives are not mentioned at any vocabulary. A model trained on report labels
inherits those gaps.

**Area Under the Margin** (Pleiss et al. 2020) tracks, for every training label,
how strongly the model agrees with it at each step. Labels the model keeps
pushing against through training are the likely-wrong ones. Here a label's
margin is its logit signed toward the teacher's side:

    margin = +logit   where the teacher says present (target >= 0.5)
             -logit   where the teacher says absent

AUM is that margin averaged over all training steps. AUM < 0 means the image
model spent the run disagreeing with the report.

**Correct, do not discard.** The textbook move is to down-weight flagged
labels. Here that would throw away exactly the cases where image and report
disagree -- the ones image-derived gold rewards. So flagged targets are moved
TOWARD the model's own (epoch-averaged) prediction instead.

**Gold-58 is never used to learn anything.** It is not in training, so it has
no AUM; flags come only from the 4349 training studies. Gold stays a pure
ruler. (Noise-transition matrices and meta-learning were rejected for exactly
this reason: they fit on the 58 and leave nothing to measure with.)

Arms, same seed and data order as the `bce` arm of `rsna-finetune-rank`:

    bce       uncorrected labels; also logs AUM. Should reproduce 0.8619
    blend     flagged targets -> 0.5*teacher + 0.5*model
    replace   flagged targets -> model

**The honest risk.** The model also learns the reports' systematic errors -- the
mechanism that made derived-CV blind. A bias the model has absorbed will not be
flagged. The gold delta says which effect wins; anything under ~0.015 is noise.
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
FLAG_FRAC = 0.10        # at most this share of each label's entries is flagged
BUDGET_H = 8.0
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
PILK = glob.glob("/kaggle/input/**/report_labels_v2.csv", recursive=True)[0]
FALLBACK = glob.glob("/kaggle/input/**/labels_rules_v5.csv", recursive=True)[0]
W = glob.glob("/kaggle/input/**/ResNet50.pt", recursive=True)[0]

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
n_slices = X.shape[1]
print(f"cache {X.shape} | {len(ids)} studies", flush=True)
assert X.shape[-1] == 224, f"expected the 224px cache, got {X.shape[-1]}px"

lab = pd.read_csv(PILK).set_index("StudyInstanceUID")[LABELS]
_fb = pd.read_csv(FALLBACK).set_index("StudyInstanceUID")[LABELS]
_missing = [u for u in ids if u not in lab.index]
if _missing:
    print(f"filling {len(_missing)} study(s) from rules_v5", flush=True)
    lab = pd.concat([lab, _fb.loc[_missing]])
G0 = lab.loc[ids].values.astype(np.float32)          # the teacher, never mutated

train_csv = pd.read_csv(os.path.join(COMP, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_mask = np.array([u in set(gold_df.index) for u in ids])
Y_gold = np.zeros((len(ids), len(LABELS)), dtype=np.int64)
Y_gold[gold_mask] = gold_df.loc[ids[gold_mask], LABELS].values.astype(int)

val = np.where(gold_mask)[0]
train = np.setdiff1d(np.arange(len(ids)), val)
assert not np.intersect1d(train, val).size, "gold leaked into training"
print(f"train {len(train)} | val {len(val)} gold", flush=True)


class Studies(Dataset):
    def __init__(self, index, targets):
        self.index, self.targets = index, targets

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = self.index[k]
        return (torch.from_numpy(np.asarray(X[i])),
                torch.from_numpy(self.targets[i]), int(i))


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
    return bb.to(dev), AttnHead(2048, len(LABELS)).to(dev)


def forward_studies(backbone, head, imgs):
    b, s, h, w = imgs.shape
    x = imgs.reshape(b * s, 1, h, w).float().div_(255.).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    return head(backbone(x).mean(dim=(2, 3)).reshape(b, s, -1))


@torch.no_grad()
def evaluate(backbone, head, targets):
    backbone.eval(); head.eval()
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    at = 0
    for imgs, _, _ in DataLoader(Studies(val, targets), batch_size=BATCH_STUDIES,
                                 shuffle=False, num_workers=2):
        with torch.amp.autocast("cuda"):
            out = forward_studies(backbone, head, imgs.to(dev, non_blocking=True))
        preds[at:at + len(out)] = out.float().cpu().numpy()
        at += len(out)
    gold = [roc_auc_score(Y_gold[val][:, j], preds[:, j])
            for j in range(len(LABELS))
            if len(np.unique(Y_gold[val][:, j])) == 2 and np.isfinite(preds[:, j]).all()]
    backbone.train(); head.train()
    return (float(np.mean(gold)) if gold else float("nan")), preds


def run_arm(name, targets, log_aum=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    backbone, head = build()
    loader = DataLoader(Studies(train, targets), batch_size=BATCH_STUDIES,
                        shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad],
         "lr": LR_BACKBONE},
        {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=EPOCHS * len(loader))
    scaler = torch.amp.GradScaler("cuda")
    bce = nn.BCEWithLogitsLoss()
    trainable = [p for p in list(backbone.parameters()) + list(head.parameters())
                 if p.requires_grad]

    # margin and sigmoid history per (epoch, study, label), training studies only
    M = np.full((EPOCHS, len(ids), len(LABELS)), np.nan, dtype=np.float32)
    PR = np.full((EPOCHS, len(ids), len(LABELS)), np.nan, dtype=np.float32)
    side = np.where(G0 >= 0.5, 1.0, -1.0).astype(np.float32)   # vs the TEACHER

    print(f"\n=== arm {name} ===\nepoch   loss     gold      time", flush=True)
    best, best_preds = -1.0, None
    t0 = time.time()
    for ep in range(EPOCHS):
        tot, k = 0.0, 0
        for imgs, tgt, idx in loader:
            imgs, tgt = imgs.to(dev, non_blocking=True), tgt.to(dev)
            with torch.amp.autocast("cuda"):
                logits = forward_studies(backbone, head, imgs)
                loss = bce(logits, tgt)
            if log_aum:
                lg = logits.detach().float().cpu().numpy()
                ii = idx.numpy()
                M[ep, ii] = lg * side[ii]
                PR[ep, ii] = 1.0 / (1.0 + np.exp(-lg))
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); k += 1
        g, preds = evaluate(backbone, head, targets)
        print(f"{ep+1:>5}{tot/max(k,1):>8.4f}{g:>9.4f}"
              f"{(time.time()-t0)/60:>9.1f}m", flush=True)
        if g > best:
            best, best_preds = g, preds.copy()
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "labels": LABELS, "size": int(X.shape[-1]),
                        "slices": int(n_slices // 3), "arch": "resnet50_radimagenet",
                        "variant": f"aum_{name}",
                        "planes": ["Sagittal", "Coronal", "Axial"], "gold": g},
                       f"/kaggle/working/finetuned_aum_{name}.pt")
    print(f"  arm {name}: best gold {best:.4f} in {(time.time()-t0)/60:.0f}m", flush=True)
    return best, best_preds, (M, PR) if log_aum else None


T0 = time.time()
results, preds = {}, {}

# ---- arm 1: control, and the AUM measurement ----
results["bce"], preds["bce"], (M, PR) = run_arm("bce", G0, log_aum=True)
arm_h = (time.time() - T0) / 3600

AUM = np.nanmean(M, axis=0)                 # [n_studies, 12]; NaN for gold rows
P_HIST = np.nanmean(PR, axis=0)             # epoch-averaged model probability
assert np.isnan(AUM[val]).all(), "gold studies have AUM -- they were trained on"
assert np.isfinite(AUM[train]).all(), "some training study was never seen"

# ---- flag ----
FLAG = np.zeros_like(G0, dtype=bool)
print(f"\nflagging: AUM < 0 and within the lowest {FLAG_FRAC:.0%} per label",
      flush=True)
print(f"  {'label':<18}{'flagged':>9}{'teacher+ ->-':>14}{'teacher- ->+':>14}"
      f"{'AUM<0 total':>13}", flush=True)
for j, labn in enumerate(LABELS):
    a = AUM[train, j]
    cap = int(FLAG_FRAC * len(train))
    order = np.argsort(a)
    pick = [o for o in order[:cap] if a[o] < 0]
    FLAG[train[pick], j] = True
    pos = G0[train[pick], j] >= 0.5
    print(f"  {labn:<18}{len(pick):>9}{pos.sum():>14}{(~pos).sum():>14}"
          f"{(a < 0).sum():>13}", flush=True)
print(f"  total flagged entries: {FLAG.sum()} of {len(train)*len(LABELS)} "
      f"({FLAG.sum()/(len(train)*len(LABELS)):.1%})", flush=True)
# The hypothesis predicts Synovitis (and other under-reported findings) shows up
# mostly in the "teacher- -> +" column: report silent, image positive.

G_blend = G0.copy()
G_blend[FLAG] = 0.5 * G0[FLAG] + 0.5 * P_HIST[FLAG]
G_repl = G0.copy()
G_repl[FLAG] = P_HIST[FLAG]

np.savez_compressed("/kaggle/working/aum.npz", study_ids=ids, aum=AUM,
                    p_hist=P_HIST, flag=FLAG, teacher=G0)
pd.DataFrame(G_blend, index=ids, columns=LABELS).rename_axis(
    "StudyInstanceUID").to_csv("/kaggle/working/labels_aum_blend.csv")

# ---- arms 2 and 3 ----
for name, tg in [("blend", G_blend), ("replace", G_repl)]:
    el = (time.time() - T0) / 3600
    if el + arm_h > BUDGET_H:
        print(f"\nSTOPPING before {name}: {el:.1f} h used", flush=True)
        break
    results[name], preds[name], _ = run_arm(name, tg)

print("\n" + "=" * 56, flush=True)
ctrl = results["bce"]
for k, v in results.items():
    d = "--" if k == "bce" else f"{v-ctrl:+.4f}"
    print(f"{k:<10}gold {v:.4f}   vs control {d}", flush=True)
print("reference: same-seed bce arm of rsna-finetune-rank scored 0.8619", flush=True)
print("Gold deltas under ~0.015 are noise (same-recipe reruns differ by ~0.008).")

ks = list(preds)
for a in range(len(ks)):
    for b in range(a + 1, len(ks)):
        c = np.mean([np.corrcoef(preds[ks[a]][:, j], preds[ks[b]][:, j])[0, 1]
                     for j in range(len(LABELS))])
        print(f"  per-label corr {ks[a]} vs {ks[b]}: {c:.3f}", flush=True)
np.savez_compressed("/kaggle/working/aum_valpreds.npz", study_ids=ids[val],
                    **{f"pred_{k}": v for k, v in preds.items()})
print("DONE", flush=True)
