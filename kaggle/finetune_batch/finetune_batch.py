"""Train four configurations and ensemble them. GPU, ~3 h.

v7 (LB 0.795) peaked at **epoch 4 of 8** on gold, then loss kept falling while
gold flattened — the model getting better at reproducing our label extractor
rather than at reading knees. But epoch 4 sat mid-schedule, with OneCycle still
annealing toward epoch 8, so the learning rate was high and the weights never
settled. Sizing the schedule to the length we actually want is the first thing
to fix, and it is nearly free.

Four configurations, each with its own seed:

    A  4 epochs, last 4 blocks   -- v7's peak, properly annealed
    B  4 epochs, last 2 blocks   -- less capacity to memorize the extractor
    C  4 epochs, last 6 blocks   -- more capacity, does it help or overfit?
    D  6 epochs, last 4 blocks   -- is 4 epochs actually enough?

Then the part we have never tried: **averaging their predictions**. Every
earlier "combination" experiment concatenated features, which is a different
operation — the models compete for PCA components. Averaging lets each keep its
own decision surface, and it is the most dependable technique in competitive
ML. All 2^4 - 1 subsets are scored so the best combination is chosen on
evidence rather than assumed to be "all of them".

Selection is on gold-58 throughout. It has now called two consecutive
leaderboard moves to within 0.001, while CV against our own extracted labels
has been wrong three times including in sign.
"""
import os
import subprocess
import sys
import time
from itertools import combinations

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
from sklearn.metrics import roc_auc_score

assert torch.cuda.is_available(), "no CUDA after install"
dev = "cuda"
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)

# (name, epochs, unfreeze_last, seed)
CONFIGS = [("A_e4_u4", 4, 4, 0),
           ("B_e4_u2", 4, 2, 1),
           ("C_e4_u6", 4, 6, 2),
           ("D_e6_u4", 6, 4, 3)]
BATCH_STUDIES = 4
LR_HEAD, LR_BACKBONE = 3e-4, 2e-5
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
assert _npy and _idx, f"cache not found: {_npy} {_idx}"
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
print(f"cache {X.shape}", flush=True)

lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
G = lab.loc[ids, LABELS].values.astype(np.float32)
train_csv = pd.read_csv(os.path.join(COMP, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)
gold_mask = np.array([u in gold_ids for u in ids])
Y_gold = np.zeros((len(ids), len(LABELS)), dtype=np.int64)
Y_gold[gold_mask] = gold_df.loc[ids[gold_mask], LABELS].values.astype(int)

# identical split for every config, so predictions can be averaged
rng = np.random.default_rng(0)
order = rng.permutation(len(ids))
val = np.unique(np.concatenate([np.where(gold_mask)[0],
                                order[:400][~gold_mask[order[:400]]]]))
train_idx = np.setdiff1d(np.arange(len(ids)), val)
print(f"train {len(train_idx)} | val {len(val)} incl. {gold_mask.sum()} gold",
      flush=True)


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


from transformers import AutoModel
ROOT = [os.path.dirname(p) for p in
        glob.glob("/kaggle/input/**/config.json", recursive=True)
        if "dinov2" in p.lower()][0]


def gold_auc(preds):
    gm = gold_mask[val]
    out = []
    for j in range(len(LABELS)):
        yg = Y_gold[val][gm, j]
        if len(np.unique(yg)) == 2:
            out.append(roc_auc_score(yg, preds[gm, j]))
    return float(np.mean(out))


def run(name, epochs, unfreeze, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone = AutoModel.from_pretrained(ROOT)
    for p in backbone.parameters():
        p.requires_grad = False
    for blk in backbone.encoder.layer[-unfreeze:]:
        for p in blk.parameters():
            p.requires_grad = True
    head = AttnHead(backbone.config.hidden_size, len(LABELS))
    backbone, head = backbone.to(dev), head.to(dev)

    def fwd(imgs):
        b, s, h, w = imgs.shape
        x = imgs.reshape(b * s, 1, h, w).float().div_(255.).repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        e = backbone(pixel_values=x).last_hidden_state[:, 0]
        return head(e.reshape(b, s, -1))

    loader = DataLoader(Studies(train_idx), batch_size=BATCH_STUDIES,
                        shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad],
         "lr": LR_BACKBONE},
        {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.BCEWithLogitsLoss()

    t = time.time()
    for ep in range(epochs):
        backbone.train(); head.train()
        for imgs, tgt in loader:
            imgs, tgt = imgs.to(dev, non_blocking=True), tgt.to(dev)
            with torch.amp.autocast("cuda"):
                loss = lossf(fwd(imgs), tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
        print(f"    {name} epoch {ep+1}/{epochs} loss {loss.item():.4f} "
              f"{(time.time()-t)/60:.0f}m", flush=True)

    backbone.eval(); head.eval()
    preds = np.zeros((len(val), len(LABELS)), dtype=np.float32)
    vl = DataLoader(Studies(val), batch_size=BATCH_STUDIES, shuffle=False,
                    num_workers=2)
    at = 0
    with torch.no_grad():
        for imgs, _ in vl:
            with torch.amp.autocast("cuda"):
                o = torch.sigmoid(fwd(imgs.to(dev, non_blocking=True)))
            preds[at:at + len(o)] = o.float().cpu().numpy()
            at += len(o)
    g = gold_auc(preds)
    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                "labels": LABELS, "size": 224, "slices": 12,
                "planes": ["Sagittal", "Coronal", "Axial"],
                "unfreeze_last": unfreeze, "epochs": epochs, "gold": g},
               f"/kaggle/working/model_{name}.pt")
    del backbone, head
    torch.cuda.empty_cache()
    return g, preds


print("\n" + "=" * 62)
print(f"{'config':<12}{'epochs':>7}{'unfrozen':>10}{'gold':>9}{'time':>8}")
print("=" * 62)
results = {}
for name, epochs, unfreeze, seed in CONFIGS:
    t = time.time()
    try:
        g, preds = run(name, epochs, unfreeze, seed)
        results[name] = preds
        print(f"{name:<12}{epochs:>7}{unfreeze:>10}{g:>9.4f}"
              f"{(time.time()-t)/60:>7.0f}m", flush=True)
    except Exception as e:
        print(f"{name:<12} FAILED: {str(e)[:80]}", flush=True)

print("\n" + "=" * 62)
print("ENSEMBLES (averaged predictions) — v7 single model was gold 0.7353")
print("=" * 62)
best, best_g = None, -1
names = list(results)
for k in range(1, len(names) + 1):
    for combo in combinations(names, k):
        g = gold_auc(np.mean([results[c] for c in combo], axis=0))
        if g > best_g:
            best, best_g = combo, g
        if k > 1 or len(names) == 1:
            print(f"  {'+'.join(combo):<40}{g:>8.4f}", flush=True)

print(f"\nBEST: {'+'.join(best)}  gold {best_g:.4f}")
np.savez_compressed("/kaggle/working/ensemble_choice.npz",
                    members=np.array(best), gold=best_g,
                    all_names=np.array(names))
print("saved checkpoints + ensemble_choice.npz\nDONE", flush=True)
