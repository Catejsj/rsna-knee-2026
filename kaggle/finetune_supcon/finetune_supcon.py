"""v26: "recognise its own kind" -- supervised contrastive training + retrieval. GPU, ~2.5 h.

The idea, in two halves:
  1. train the embedding so knees with the same findings sit together
     (supervised contrastive learning);
  2. use the training set as a searchable database: a test knee retrieves its
     nearest training knees and predicts from their labels.

Measured already (2026-10-05): retrieval in UNTRAINED feature spaces loses to a
learned readout by 0.04-0.13 -- generic kNN, slice matching, Optimal Transport
and graph label propagation all retrieve knees scanned alike, not sick alike.
This tests the version with a LEARNED metric, which is the only one left.

Arms (same seed, same data order, v19B recipe -- RadImageNet R50 layer3+4,
pilkwang teacher, 4349 studies, gold-58 held out):

    bce      plain recipe; reproduces the 0.8619 control and gives retrieval on
             FINE-TUNED embeddings (the middle step between frozen and contrastive)
    supcon   BCE + lambda * soft supervised-contrastive loss

Multi-label similarity. Two knees are "the same kind" in proportion to how
similar their 12 graded teacher labels are: target distribution over the
memory bank = softmax(cos(label_i, label_k) / TAU_LABEL). Partial overlap is
handled naturally, and all-normal knees have meaningful neighbours too -- a
hard Jaccard would leave them undefined.

Batch is 2 studies, so contrastive pairs come from a detached MEMORY BANK of the
last BANK studies (MoCo-style), the same fix that made the ranking-loss test
measure something. The projection head is created inside `fork_rng` so its
initialisation cannot shift the data order relative to the 0.8619 control.

Retrieval is evaluated three ways after training, database = training studies,
queries = gold, labels = teacher, kernel-weighted soft vote, best over a k/tau
grid (optimistic in retrieval's favour):
    pooled     mean of slice features (2048-d)
    proj       the contrastive space (128-d; supcon arm only)
    per-label  label j retrieves in the head's attention-pooled vector for j
"""
import os
import subprocess
import sys
import time

print("installing torch 2.6.0+cu118 (identical stack to the 0.8619 control)", flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.6.0", "torchvision==0.21.0",
                    "--index-url", "https://download.pytorch.org/whl/cu118"],
                   capture_output=True, text=True)
print(f"pip rc={r.returncode}", flush=True)
if r.returncode != 0:
    print((r.stdout + r.stderr)[-1200:]); sys.exit("install failed")

import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchvision
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset

assert torch.cuda.is_available()
dev = "cuda"
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)

EPOCHS, BATCH_STUDIES = 8, 2
LR_HEAD, LR_BACKBONE = 3e-4, 2e-5
SEED = 0
LAMBDA = 0.5          # weight of the contrastive term
TAU = 0.1             # embedding temperature
TAU_LABEL = 0.1       # sharpness of the label-similarity target
BANK = 512
PROJ = 128
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
assert X.shape[-1] == 224
lab = pd.read_csv(PILK).set_index("StudyInstanceUID")[LABELS]
_fb = pd.read_csv(FALLBACK).set_index("StudyInstanceUID")[LABELS]
_missing = [u for u in ids if u not in lab.index]
if _missing:
    lab = pd.concat([lab, _fb.loc[_missing]])
G = lab.loc[ids].values.astype(np.float32)
Gt = torch.from_numpy(G).to(dev)
tr = pd.read_csv(os.path.join(COMP, "train.csv"))
gold_df = tr[tr["ACL"].notna()].set_index("StudyInstanceUID")
gold_mask = np.array([u in set(gold_df.index) for u in ids])
Y_gold = np.zeros((len(ids), len(LABELS)), dtype=np.int64)
Y_gold[gold_mask] = gold_df.loc[ids[gold_mask], LABELS].values.astype(int)
val = np.where(gold_mask)[0]
train = np.setdiff1d(np.arange(len(ids)), val)
Yv = Y_gold[val]
print(f"train {len(train)} | gold {len(val)}", flush=True)


class Studies(Dataset):
    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = self.index[k]
        return torch.from_numpy(np.asarray(X[i])), torch.from_numpy(G[i]), int(i)


class AttnHead(nn.Module):
    def __init__(self, dim, n_out, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(n_out, dim) * dim ** -0.5)
        self.value = nn.Linear(dim, hidden)
        self.out = nn.Parameter(torch.zeros(n_out, hidden))
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.normal_(self.out, std=hidden ** -0.5)

    def pooled(self, emb):
        e = self.norm(emb)
        attn = torch.einsum("bsd,cd->bcs", e, self.query) * e.shape[-1] ** -0.5
        return torch.einsum("bcs,bsh->bch", attn.softmax(-1), self.value(e))

    def forward(self, emb):
        return (self.pooled(emb) * self.out).sum(-1) + self.bias


def build():
    net = torchvision.models.resnet50(weights=None)
    bb = nn.Sequential(*list(net.children())[:-2])
    sd = torch.load(W, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    bb.load_state_dict(sd, strict=True)
    for p in bb.parameters():
        p.requires_grad = False
    for p in list(bb[7].parameters()) + list(bb[6].parameters()):
        p.requires_grad = True
    return bb.to(dev), AttnHead(2048, len(LABELS)).to(dev)


def slice_feats(backbone, imgs):
    b, s, h, w = imgs.shape
    x = imgs.reshape(b * s, 1, h, w).float().div_(255.).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    return backbone(x).mean(dim=(2, 3)).reshape(b, s, -1)          # [b, s, 2048]


def soft_supcon(z, y, bank_z, bank_y):
    """Cross-entropy between embedding similarities and label similarities.

    z [B,P] (L2-normalised, with grad), y [B,12]; bank_* detached history.
    Computed in fp32 -- fp16 overflow on similarity logits is the failure mode
    that produced NaNs with EfficientNet on this hardware.
    """
    z, y = z.float(), y.float()
    logits = z @ bank_z.T / TAU                                     # [B, N]
    ly = Fn.normalize(y, dim=1) @ Fn.normalize(bank_y, dim=1).T     # [B, N]
    target = (ly / TAU_LABEL).softmax(1)
    return -(target * logits.log_softmax(1)).sum(1).mean()


def _selftest():
    """The loss must prefer an embedding that mirrors label similarity."""
    y = torch.tensor([[1., 0, 0], [0, 1, 0]])
    by = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
    bz = Fn.normalize(torch.eye(3), dim=1)
    good = soft_supcon(Fn.normalize(torch.tensor([[1., 0, 0], [0, 1, 0]]), dim=1), y, bz, by)
    bad = soft_supcon(Fn.normalize(torch.tensor([[0., 1, 0], [1, 0, 0]]), dim=1), y, bz, by)
    assert good < bad, f"contrastive loss does not prefer the aligned embedding ({good} vs {bad})"
    print(f"  soft_supcon selftest ok (aligned {good:.3f} < swapped {bad:.3f})", flush=True)


_selftest()


def macro(P, Y):
    out = []
    for j in range(len(LABELS)):
        y = Y[:, j]
        if len(np.unique(y)) == 2 and np.isfinite(P[:, j]).all():
            r = rankdata(P[:, j]); n1 = y.sum(); n0 = len(y) - n1
            out.append((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out))


@torch.no_grad()
def extract(backbone, head, proj):
    """Logits, pooled 2048-d, projection, and per-label pooled vectors for ALL studies."""
    backbone.eval(); head.eval()
    if proj is not None:
        proj.eval()
    n = len(ids)
    LOG = np.zeros((n, 12), np.float32); POOL = np.zeros((n, 2048), np.float32)
    PER = np.zeros((n, 12, 256), np.float32)
    PRJ = np.zeros((n, PROJ), np.float32) if proj is not None else None
    for imgs, _, idx in DataLoader(Studies(np.arange(n)), batch_size=4, num_workers=2):
        with torch.amp.autocast("cuda"):
            f = slice_feats(backbone, imgs.to(dev))
        f = f.float()
        ii = idx.numpy()
        LOG[ii] = head(f).cpu().numpy()
        PER[ii] = head.pooled(f).cpu().numpy()
        pooled = f.mean(1)
        POOL[ii] = pooled.cpu().numpy()
        if proj is not None:
            PRJ[ii] = Fn.normalize(proj(pooled), dim=1).cpu().numpy()
    backbone.train(); head.train()
    if proj is not None:
        proj.train()
    return LOG, POOL, PER, PRJ


def knn_vote(Zdb, Zq, Gdb, k, tau):
    Zdb = Zdb / np.maximum(np.linalg.norm(Zdb, axis=1, keepdims=True), 1e-9)
    Zq = Zq / np.maximum(np.linalg.norm(Zq, axis=1, keepdims=True), 1e-9)
    S = Zq @ Zdb.T
    top = np.argsort(-S, 1)[:, :k]
    s = np.take_along_axis(S, top, 1)
    w = np.exp((s - s.max(1, keepdims=True)) / tau); w /= w.sum(1, keepdims=True)
    return np.einsum("qk,qk...->q...", w, Gdb[top])


GRID = [(k, t) for k in (10, 25, 50, 100, 200) for t in (0.02, 0.05, 0.2)]


def retrieval_report(name, LOG, POOL, PER, PRJ):
    head_p = LOG[val]
    head_g = macro(head_p, Yv)
    rows = {"head (trained readout)": (head_g, head_p)}
    best = max(((macro(knn_vote(POOL[train], POOL[val], G[train], k, t), Yv), k, t)
                for k, t in GRID))
    rows["kNN pooled 2048-d"] = (best[0], knn_vote(POOL[train], POOL[val], G[train], best[1], best[2]))
    if PRJ is not None:
        b2 = max(((macro(knn_vote(PRJ[train], PRJ[val], G[train], k, t), Yv), k, t)
                  for k, t in GRID))
        rows["kNN contrastive 128-d"] = (b2[0], knn_vote(PRJ[train], PRJ[val], G[train], b2[1], b2[2]))
    # per-label: label j retrieves in the head's pooled vector for label j
    best_pl = (-1, None)
    for k, t in GRID:
        P = np.stack([knn_vote(PER[train][:, j], PER[val][:, j], G[train][:, j:j+1], k, t)[:, 0]
                      for j in range(12)], 1)
        a = macro(P, Yv)
        if a > best_pl[0]:
            best_pl = (a, P)
    rows["kNN per-label space"] = best_pl
    print(f"\n  [{name}] retrieval vs trained readout (gold, kNN best over k/tau)", flush=True)
    for r_, (g, P) in rows.items():
        c = np.mean([np.corrcoef(P[:, j], head_p[:, j])[0, 1] for j in range(12)])
        print(f"    {r_:<26}{g:.4f}   corr w/ head {c:.3f}", flush=True)
    # does a retrieval member help the head? probability-scale average
    hp = 1 / (1 + np.exp(-head_p))
    for r_ in [k for k in rows if k.startswith("kNN")]:
        print(f"    head + {r_:<19} avg {macro((hp + rows[r_][1]) / 2, Yv):.4f}", flush=True)
    return {k: v[0] for k, v in rows.items()}


def run_arm(name, use_con):
    torch.manual_seed(SEED); np.random.seed(SEED)
    backbone, head = build()
    proj = None
    if use_con:
        # its own RNG stream: must not shift the data order vs the control
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12345)
            proj = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, PROJ)).to(dev)
    loader = DataLoader(Studies(train), batch_size=BATCH_STUDIES, shuffle=True,
                        num_workers=2, drop_last=True)
    params = [{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
              {"params": head.parameters(), "lr": LR_HEAD}]
    mx = [LR_BACKBONE, LR_HEAD]
    if proj is not None:
        params.append({"params": proj.parameters(), "lr": LR_HEAD}); mx.append(LR_HEAD)
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=mx, total_steps=EPOCHS * len(loader))
    scaler = torch.amp.GradScaler("cuda")
    bce = nn.BCEWithLogitsLoss()
    trainable = [p for g_ in params for p in g_["params"] if p.requires_grad]
    bank_z = torch.zeros(BANK, PROJ, device=dev); bank_y = torch.zeros(BANK, 12, device=dev)
    bank_n, bank_at = 0, 0

    print(f"\n=== arm {name} ===\nepoch   bce     con      gold     time", flush=True)
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(EPOCHS):
        tb = tc = 0.0; k = 0
        for imgs, tgt, idx in loader:
            imgs, tgt = imgs.to(dev, non_blocking=True), tgt.to(dev)
            with torch.amp.autocast("cuda"):
                f = slice_feats(backbone, imgs)
                lb = bce(head(f), tgt)
            lc = torch.zeros((), device=dev)
            if proj is not None:
                z = Fn.normalize(proj(f.float().mean(1)), dim=1)
                if bank_n >= 32:
                    lc = soft_supcon(z, tgt, bank_z[:bank_n], bank_y[:bank_n])
            loss = lb + LAMBDA * lc
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            if proj is not None:
                for zz, yy in zip(z.detach(), tgt.detach()):
                    bank_z[bank_at] = zz; bank_y[bank_at] = yy
                    bank_at = (bank_at + 1) % BANK; bank_n = min(bank_n + 1, BANK)
            tb += lb.item(); tc += float(lc); k += 1
        # gold on the head, every epoch (same selection rule as every member)
        backbone.eval(); head.eval()
        with torch.no_grad():
            P = []
            for imgs, _, _ in DataLoader(Studies(val), batch_size=2, num_workers=2):
                with torch.amp.autocast("cuda"):
                    P.append(head(slice_feats(backbone, imgs.to(dev))).float().cpu().numpy())
        backbone.train(); head.train()
        g = macro(np.concatenate(P), Yv)
        print(f"{ep+1:>5}{tb/max(k,1):>8.4f}{tc/max(k,1):>8.4f}{g:>9.4f}"
              f"{(time.time()-t0)/60:>8.1f}m", flush=True)
        if g > best:
            best = g
            best_state = ({k_: v.detach().clone() for k_, v in backbone.state_dict().items()},
                          {k_: v.detach().clone() for k_, v in head.state_dict().items()},
                          None if proj is None else {k_: v.detach().clone() for k_, v in proj.state_dict().items()})
    print(f"  arm {name}: best gold {best:.4f}", flush=True)
    backbone.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    if proj is not None:
        proj.load_state_dict(best_state[2])
    torch.save({"backbone": best_state[0], "head": best_state[1], "labels": LABELS,
                "size": 224, "slices": 12, "arch": "resnet50_radimagenet",
                "variant": f"supcon_{name}", "planes": ["Sagittal", "Coronal", "Axial"],
                "gold": best}, f"/kaggle/working/finetuned_supcon_{name}.pt")
    out = extract(backbone, head, proj)
    np.savez_compressed(f"/kaggle/working/supcon_{name}_feats.npz", study_ids=ids,
                        logits=out[0], pooled=out[1].astype(np.float16),
                        **({} if out[3] is None else {"proj": out[3]}))
    return best, retrieval_report(name, *out)


res = {}
for name, con in (("bce", False), ("supcon", True)):
    res[name] = run_arm(name, con)

print("\n" + "=" * 60, flush=True)
print(f"head gold:  bce {res['bce'][0]:.4f} (control should be ~0.8619) | "
      f"supcon {res['supcon'][0]:.4f} | delta {res['supcon'][0]-res['bce'][0]:+.4f}", flush=True)
print("gold deltas under ~0.015 are noise (same-recipe reruns differ by ~0.008).")
print("DONE", flush=True)
