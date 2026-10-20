"""v22: does training the METRIC beat training the likelihood? GPU, ~3 h (3 arms).

The competition scores **macro-averaged AUC**, which depends only on the ORDER of
the predictions. Every model in this project has been trained with
`BCEWithLogitsLoss`, which optimises calibration -- it cares how close each
probability is to its target, not how the studies rank against each other. A
perfectly calibrated model and a perfectly ranked one are different objectives,
and we have only ever asked for the first.

This is the last untried axis on the exhaust list, and unlike the others it costs
**zero inference time**: the loss exists only during training, so whatever it buys
is free against the 9 h submission cap.

### Why a naive pairwise loss would measure nothing here

`BATCH_STUDIES = 2`. A pairwise ranking loss computed inside the batch sees ONE
pair per label, and for most labels both studies have the same target, so it sees
none. Any result would be an artifact of the batch size, not of the idea.

Fix: a **detached FIFO buffer** of the last `BUF_CAP` studies' logits. The current
batch is ranked against the buffer, so each step forms up to `2 x 256 x 12` ordered
pairs instead of one. Only the current batch carries gradient; the buffer is
`.detach()`ed history. Standard practice for AUC surrogates under small batches.
The buffer is stale by up to ~128 optimiser steps, which at `LR_BACKBONE=2e-5` is
a slow-moving model -- acceptable, and the alternative (batch 2) measures nothing.

### What counts as an ordered pair

The teacher emits five graded levels: **0.08 / 0.28 / 0.68 / 0.82 / 0.94**
(the rules_v5 fallback uses 0.08 / 0.25 / 0.65 / 0.90 -- same shape). Adjacent
gaps are 0.20, 0.40, 0.14, 0.12.

`PAIR_DELTA = 0.30` therefore admits **exactly the cross-boundary pairs**:
{0.68, 0.82, 0.94} ranked above {0.08, 0.28}. That is precisely the binary AUC
the competition computes. It deliberately does NOT try to rank 0.94 above 0.82 --
those are the teacher's confidence gradations, they are noisy, and the metric
gives no credit for getting them in order. The graded information is not thrown
away: the BCE term in the hybrid arm still sees the full graded target.

### Arms (single-variable; same seed, same split, same data order)

    bce      rank_w 0.0            pure BCE -- the control
    hybrid   rank_w 1.0 + BCE      calibration AND ranking
    rank     rank_w 1.0, no BCE    ranking only

The **control is the honest part of this experiment**. v19B scored gold 0.8700 but
its torch RNG was never seeded, so it is not bit-reproducible; `bce` re-runs it
under a fixed seed and is the number the other two arms are compared against.
Do not compare `hybrid` to 0.8700.

`rank` is expected to score well and ensemble badly: with no BCE term the logits
are only defined up to a monotone transform, so `sigmoid(logit)` stops being a
probability. Our ensemble is a **probability average**, and rank-averaging has
already been measured as a loss (-0.0015). If `rank` wins alone but its
predictions do not average, that is the interesting finding, not a failure.

Recipe is otherwise v19B verbatim (RadImageNet R50, layer3+4, pilkwang teacher,
4349 studies, gold-58 held out) so the loss is the only variable.
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
BUF_CAP = 256          # studies of detached ranking history
PAIR_DELTA = 0.30      # see header: admits exactly the cross-boundary pairs
MARGIN = 1.0           # logit margin the hinge demands
BUDGET_H = 8.0         # GPU kernels are capped at 9 h
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
print(f"weights: {W}", flush=True)

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
n_slices = X.shape[1]
print(f"cache {X.shape} | {len(ids)} studies", flush=True)
assert X.shape[-1] == 224, (
    f"expected the 224px cache, got {X.shape[-1]}px. Both caches are written as "
    f"slices_224_12x3.npy, so only the shape is trustworthy.")

lab = pd.read_csv(PILK).set_index("StudyInstanceUID")[LABELS]
_fb = pd.read_csv(FALLBACK).set_index("StudyInstanceUID")[LABELS]
_missing = [u for u in ids if u not in lab.index]
if _missing:
    print(f"filling {len(_missing)} study(s) from rules_v5", flush=True)
    lab = pd.concat([lab, _fb.loc[_missing]])
G = lab.loc[ids].values.astype(np.float32)

# The pairing threshold is only meaningful against the teacher's actual scale, so
# assert it rather than trusting the header. A future teacher on a 0/1 scale would
# still work; one on a compressed scale would silently form no pairs at all.
_lv = np.unique(G)
print(f"teacher levels: {np.round(_lv, 3)}", flush=True)
_span = float(_lv.max() - _lv.min())
assert _span > PAIR_DELTA, (
    f"teacher spans only {_span:.2f} but PAIR_DELTA is {PAIR_DELTA} -- no ordered "
    f"pair could ever form and the ranking arms would silently reduce to BCE.")

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


# ---- the ranking loss -------------------------------------------------------
class RankBuffer:
    """FIFO of detached (logit, target) rows so a batch of 2 can form real pairs."""

    def __init__(self, cap, n_lab, device):
        self.s = torch.zeros(cap, n_lab, device=device)
        self.t = torch.zeros(cap, n_lab, device=device)
        self.cap, self.n, self.at = cap, 0, 0

    def add(self, logits, tgt):
        b = logits.shape[0]
        for k in range(b):
            self.s[self.at] = logits[k].detach().float()
            self.t[self.at] = tgt[k].detach().float()
            self.at = (self.at + 1) % self.cap
            self.n = min(self.n + 1, self.cap)


def rank_loss(logits, tgt, buf):
    """Squared-hinge pairwise AUC surrogate, current batch vs detached history.

    Computed in fp32 on purpose. The squared hinge on raw logits is exactly the
    kind of term that overflowed EfficientNet under fp16 on sm_60 (no bf16), and
    an inf here would poison the GradScaler for the whole run.
    """
    if buf.n == 0:
        return logits.sum() * 0.0
    s = logits.float()
    t = tgt.float()
    S, T = buf.s[:buf.n], buf.t[:buf.n]
    d_t = t.unsqueeze(1) - T.unsqueeze(0)          # [B, N, L] target gap
    d_s = s.unsqueeze(1) - S.unsqueeze(0)          # [B, N, L] logit gap
    above = (d_t > PAIR_DELTA).float()             # batch item must rank higher
    below = (d_t < -PAIR_DELTA).float()            # batch item must rank lower
    la = torch.clamp(MARGIN - d_s, min=0).pow(2) * above
    lb = torch.clamp(MARGIN + d_s, min=0).pow(2) * below
    # clamp(min=1) not .item(): when no pair exists la+lb is already 0, so 0/1 = 0
    # and we avoid a device sync on every one of the ~52k training steps.
    return (la.sum() + lb.sum()) / (above.sum() + below.sum()).clamp(min=1.0)


def _selftest():
    """Three checks. A ranking loss that silently returns ~0 would look like a
    clean null result rather than a bug, which is the failure mode this project
    keeps hitting."""
    b = RankBuffer(4, 1, "cpu")
    b.s[:2] = torch.tensor([[1.0], [-1.0]])
    b.t[:2] = torch.tensor([[0.94], [0.08]])
    b.n = 2
    tg = torch.tensor([[0.94], [0.08]])

    perfect = rank_loss(torch.tensor([[1.0], [-1.0]]), tg, b).item()
    assert perfect < 1e-6, f"correct ordering must cost nothing, got {perfect}"

    inverted = rank_loss(torch.tensor([[-1.0], [1.0]]), tg, b).item()
    assert inverted > 0.9, f"inverted ordering must be penalised, got {inverted}"

    # Ordered CORRECTLY but by less than MARGIN -- the hinge must still bite,
    # otherwise this is plain sign agreement and the margin does nothing.
    # Both pairs separate by 0.5 against a margin of 1.0, so (1-0.5)^2 = 0.25.
    # (The first version of this check separated them by 1.75, i.e. past the
    # margin, and correctly returned 0 -- the test was wrong, not the loss.)
    thin = rank_loss(torch.tensor([[-0.5], [0.5]]), tg, b).item()
    assert 0.0 < thin < inverted, f"margin inactive, got {thin}"

    # and a pair must not form inside the positive grades (0.94 vs 0.82)
    b2 = RankBuffer(4, 1, "cpu")
    b2.s[:1] = torch.tensor([[-5.0]])
    b2.t[:1] = torch.tensor([[0.82]])
    b2.n = 1
    same = rank_loss(torch.tensor([[-5.0]]), torch.tensor([[0.94]]), b2).item()
    assert same == 0.0, f"0.94 vs 0.82 must not be an ordered pair, got {same}"
    print(f"  rank_loss selftest ok (perfect {perfect:.3f} | thin {thin:.3f} | "
          f"inverted {inverted:.3f})", flush=True)


_selftest()

# ---- split ------------------------------------------------------------------
rng = np.random.default_rng(0)
val = np.where(gold_mask)[0]
train = np.setdiff1d(np.arange(len(ids)), val)
assert not np.intersect1d(train, np.where(gold_mask)[0]).size, \
    "gold studies leaked into training -- the ruler would be worthless"
print(f"train {len(train)} | val {len(val)} incl. {gold_mask.sum()} gold", flush=True)


def build():
    """Fresh RadImageNet R50 + head, layer3+4 trainable (the v19B/v17B recipe)."""
    net = torchvision.models.resnet50(weights=None)
    bb = nn.Sequential(*list(net.children())[:-2])          # -> [B, 2048, 7, 7]
    sd = torch.load(W, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    bb.load_state_dict(sd, strict=True)                     # strict: no silent misses
    for p in bb.parameters():
        p.requires_grad = False
    for p in bb[7].parameters():                            # layer4
        p.requires_grad = True
    for p in bb[6].parameters():                            # layer3
        p.requires_grad = True
    hd = AttnHead(2048, len(LABELS))
    return bb.to(dev), hd.to(dev)


def forward_studies(backbone, head, imgs):
    b, s, h, w = imgs.shape
    x = imgs.reshape(b * s, 1, h, w).float().div_(255.).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    f = backbone(x).mean(dim=(2, 3))                        # global average pool
    return head(f.reshape(b, s, -1))


@torch.no_grad()
def evaluate(backbone, head):
    backbone.eval(); head.eval()
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val), batch_size=BATCH_STUDIES, shuffle=False,
                    num_workers=2)
    at = 0
    for imgs, _ in vl:
        with torch.amp.autocast("cuda"):
            out = forward_studies(backbone, head, imgs.to(dev, non_blocking=True))
        preds[at:at + len(out)] = out.float().cpu().numpy()
        at += len(out)
    gm = gold_mask[val]
    gold, derived = [], []
    for j in range(len(LABELS)):
        yb = (G[val][:, j] >= 0.5).astype(int)
        if len(np.unique(yb)) == 2 and np.isfinite(preds[:, j]).all():
            derived.append(roc_auc_score(yb, preds[:, j]))
        yg = Y_gold[val][gm, j]
        if len(np.unique(yg)) == 2 and np.isfinite(preds[gm, j]).all():
            gold.append(roc_auc_score(yg, preds[gm, j]))
    backbone.train(); head.train()
    d = float(np.mean(derived)) if derived else float("nan")
    g = float(np.mean(gold)) if gold else float("nan")
    return d, g, preds


def run_arm(name, rank_w, use_bce):
    # Same seed for every arm: identical head init and identical data order, so
    # the loss is the only difference between the three numbers.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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
    bce = nn.BCEWithLogitsLoss()
    buf = RankBuffer(BUF_CAP, len(LABELS), dev)

    print(f"\n=== arm {name}: rank_w={rank_w} bce={use_bce} ===", flush=True)
    print("epoch   loss    bce     rank   derived   gold      time", flush=True)
    best, best_preds, skipped = -1.0, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        backbone.train(); head.train()
        tot = tb = tr = 0.0
        k = 0
        for imgs, tgt in loader:
            imgs, tgt = imgs.to(dev, non_blocking=True), tgt.to(dev)
            with torch.amp.autocast("cuda"):
                logits = forward_studies(backbone, head, imgs)
                lb = bce(logits, tgt)
            lr_ = rank_loss(logits, tgt, buf) if rank_w > 0 else logits.sum() * 0.0
            loss = (lb if use_bce else lb * 0.0) + rank_w * lr_
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                buf.add(logits, tgt)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(backbone.parameters()) + list(head.parameters())
                 if p.requires_grad], 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            buf.add(logits, tgt)
            tot += loss.item(); tb += lb.item(); tr += float(lr_); k += 1
        d, g, preds = evaluate(backbone, head)
        k = max(k, 1)
        print(f"{ep+1:>5}{tot/k:>8.4f}{tb/k:>8.4f}{tr/k:>8.4f}"
              f"{d:>10.4f}{g:>8.4f}{(time.time()-t0)/60:>9.1f}m", flush=True)
        if g > best:
            best, best_preds = g, preds.copy()
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "labels": LABELS, "size": int(X.shape[-1]),
                        "slices": int(n_slices // 3), "arch": "resnet50_radimagenet",
                        "variant": f"rank_{name}", "rank_w": rank_w,
                        "use_bce": use_bce,
                        "planes": ["Sagittal", "Coronal", "Axial"], "gold": g},
                       f"/kaggle/working/finetuned_rank_{name}.pt")
    if skipped:
        print(f"  WARNING: skipped {skipped} non-finite steps", flush=True)
    print(f"  arm {name}: best gold {best:.4f} in {(time.time()-t0)/60:.0f}m",
          flush=True)
    return best, best_preds


ARMS = [("bce", 0.0, True), ("hybrid", 1.0, True), ("rank", 1.0, False)]
results, allpreds = {}, {}
T0 = time.time()
for i, (name, w, ub) in enumerate(ARMS):
    el = (time.time() - T0) / 3600
    if i and el + (el / i) > BUDGET_H:
        print(f"\nSTOPPING after {i} arm(s): {el:.1f} h used, one more would "
              f"pass the {BUDGET_H} h budget. Remaining: "
              f"{[a[0] for a in ARMS[i:]]}", flush=True)
        break
    b, p = run_arm(name, w, ub)
    results[name], allpreds[name] = b, p

print("\n" + "=" * 60, flush=True)
print("arm       gold     vs bce control", flush=True)
ctrl = results.get("bce")
for k, v in results.items():
    delta = f"{v - ctrl:+.4f}" if ctrl is not None and k != "bce" else "--"
    print(f"{k:<10}{v:.4f}   {delta}", flush=True)

# Correlation between arms decides whether a winner is also an ensemble member.
# Being different beats being good -- but only for members that are also
# competitive, so print both numbers together.
if len(allpreds) > 1:
    print("\ncorrelation of val predictions between arms:", flush=True)
    ks = list(allpreds)
    for a in range(len(ks)):
        for b_ in range(a + 1, len(ks)):
            c = np.corrcoef(allpreds[ks[a]].ravel(), allpreds[ks[b_]].ravel())[0, 1]
            print(f"  {ks[a]:<8} vs {ks[b_]:<8} {c:.3f}", flush=True)

np.savez_compressed("/kaggle/working/rank_valpreds.npz",
                    study_ids=ids[val],
                    **{f"pred_{k}": v for k, v in allpreds.items()})
print(f"\ngold-58 reference: v19B (same recipe, unseeded) 0.8700", flush=True)
print("DONE", flush=True)
