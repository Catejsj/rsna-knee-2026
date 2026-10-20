"""v25: MedImageInsight as a frozen ensemble member. GPU (T4), ~2-3 h.

The only untried PRETRAINING DOMAIN, which is the axis that bought RadImageNet
+0.016 -- the biggest ensemble gain of the project. Microsoft's model, MIT,
research use; weights from `lion-ai/MedImageInsights`.

Why FROZEN. DaViT (dims 256/512/1024/2048, depths 1/1/9/1) at 480x480 is
~540 GFLOPs/slice, ~12x DINOv2-base. Fine-tuning would cost ~9 h per epoch.
So: extract one 2048-d embedding per slice once, then train the same
per-diagnosis attention head every other member uses. Frozen features have
been weak here (fine-tuning was worth +0.053 at v7), so the question this
answers is COMPETITIVENESS: a member must land within ~0.01 of the best
(v19B 0.8700) to help, however decorrelated it is.

Vision encoder only. `davit_v1.py` + `registry.py` are copied into a private
package so nothing imports the text encoder or its transformers==4.16 pin.
Weights load non-strictly (the checkpoint also holds the language model) with a
hard assert that NO encoder key is missing -- a silent partial load would look
like a weak model rather than a bug.

Input: the existing 224px cache upsampled to 480. That costs no decode, but it
also means MedImageInsight never sees detail beyond 224. A negative result is
therefore "frozen MII on our 224 cache", not "MII at native resolution".
"""
import glob
import os
import shutil
import subprocess
import sys
import time

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mup==1.0.0",
                    "einops", "huggingface_hub"], capture_output=True, text=True)
print(f"pip rc={r.returncode}", flush=True)
if r.returncode != 0:
    print((r.stdout + r.stderr)[-1500:]); sys.exit("install failed")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import yaml
from huggingface_hub import snapshot_download
from sklearn.metrics import roc_auc_score

dev = "cuda"
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
SIZE_IN = 480
BUDGET_H = 6.5          # extraction must leave room for head training in 9 h
CHUNK = 12

# ---- model ------------------------------------------------------------------
repo = snapshot_download("lion-ai/MedImageInsights",
                         allow_patterns=["MedImageInsight/ImageEncoder/*",
                                         "2024.09.27/config.yaml",
                                         "2024.09.27/vision_model/*"])
pkg = "/kaggle/working/mii_vis"
os.makedirs(pkg, exist_ok=True)
for f in ("davit_v1.py", "registry.py"):
    shutil.copy(f"{repo}/MedImageInsight/ImageEncoder/{f}", pkg)
open(f"{pkg}/__init__.py", "w").close()
sys.path.insert(0, "/kaggle/working")
from mii_vis.davit_v1 import image_encoder      # noqa: E402

cfg = yaml.safe_load(open(f"{repo}/2024.09.27/config.yaml"))
enc_cfg = cfg["IMAGE_ENCODER"] if "IMAGE_ENCODER" in cfg else cfg["MODEL"]["IMAGE_ENCODER"]
enc_cfg = dict(enc_cfg, LOAD_PRETRAINED=False)
assert list(enc_cfg["IMAGE_SIZE"]) == [SIZE_IN, SIZE_IN], enc_cfg["IMAGE_SIZE"]
enc = image_encoder(enc_cfg, verbose=False)

ckpt = glob.glob(f"{repo}/2024.09.27/vision_model/*.pt")[0]
# Despite the .pt extension this is a SAFETENSORS file (header confirmed:
# 8-byte length, then JSON with "image_encoder.*" keys). torch.load fails on it
# with "invalid load key '\x10'"; the repo's own loader uses load_file too.
from safetensors.torch import load_file
raw = load_file(ckpt)
keys = list(raw)
print(f"checkpoint: {len(keys)} tensors; prefixes "
      f"{sorted({k.split('.')[0] for k in keys})}", flush=True)
sd = {}
for k, v in raw.items():
    k2 = k[len("module."):] if k.startswith("module.") else k
    if k2.startswith("image_encoder."):
        sd[k2[len("image_encoder."):]] = v
missing, unexpected = enc.load_state_dict(sd, strict=False)
missing = [m for m in missing if not m.startswith("head")]   # head = Identity
print(f"encoder: loaded {len(sd)} tensors | missing {len(missing)} | "
      f"unexpected {len(unexpected)}", flush=True)
assert not missing, f"encoder keys missing -- partial load: {missing[:8]}"
del raw
enc.enable_checkpoint = False          # activation checkpointing is for training
enc = enc.to(dev).eval().half()
nparam = sum(p.numel() for p in enc.parameters())
print(f"vision encoder {nparam/1e6:.0f}M params, fp16", flush=True)
torch.save({k: v.half() for k, v in enc.state_dict().items()},
           "/kaggle/working/mii_vision_fp16.pt")     # for an offline submission

MEAN = torch.tensor(enc_cfg["IMAGE_MEAN"], device=dev).view(1, 3, 1, 1).half()
STD = torch.tensor(enc_cfg["IMAGE_STD"], device=dev).view(1, 3, 1, 1).half()

# ---- data -------------------------------------------------------------------
_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
assert X.shape[-1] == 224, f"expected the 224px cache, got {X.shape[-1]}"
n, S = X.shape[0], X.shape[1]
print(f"cache {X.shape}", flush=True)


@torch.no_grad()
def embed_study(i):
    x = torch.from_numpy(np.asarray(X[i])).to(dev).half().div_(255.)
    x = Fn.interpolate(x.unsqueeze(1), size=(SIZE_IN, SIZE_IN), mode="bilinear",
                       align_corners=False).repeat(1, 3, 1, 1)
    x = (x - MEAN) / STD
    out = [enc.forward_features(x[c:c + CHUNK]) for c in range(0, S, CHUNK)]
    return torch.cat(out).float().cpu().numpy()


# throughput probe: local estimates have been 3.5x optimistic four times
_t = time.time()
for i in range(10):
    e = embed_study(i)
torch.cuda.synchronize()
per = (time.time() - _t) / 10
print(f"PROBE: {per:.2f} s/study -> {per*n/3600:.1f} h for {n} studies "
      f"| emb {e.shape}, finite {np.isfinite(e).all()}, std {e.std():.3f}", flush=True)
if per * n / 3600 > BUDGET_H:
    raise SystemExit(f"projected {per*n/3600:.1f} h > {BUDGET_H} h budget")
assert np.isfinite(e).all() and e.std() > 1e-3, "degenerate embeddings"

E = np.lib.format.open_memmap("/kaggle/working/mii_embeddings.npy", mode="w+",
                              dtype=np.float16, shape=(n, S, e.shape[1]))
t0 = time.time()
for i in range(n):
    E[i] = embed_study(i)
    if (i + 1) % 500 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{n} {el/(i+1):.2f} s/study eta {el/(i+1)*(n-i-1)/60:.0f} min",
              flush=True)
E.flush()
print(f"embedded {n} studies in {(time.time()-t0)/3600:.2f} h", flush=True)
np.save("/kaggle/working/mii_study_ids.npy", ids)

# ---- head -------------------------------------------------------------------
PILK = glob.glob("/kaggle/input/**/report_labels_v2.csv", recursive=True)[0]
FALLBACK = glob.glob("/kaggle/input/**/labels_rules_v5.csv", recursive=True)[0]
COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
lab = pd.read_csv(PILK).set_index("StudyInstanceUID")[LABELS]
fb = pd.read_csv(FALLBACK).set_index("StudyInstanceUID")[LABELS]
miss = [u for u in ids if u not in lab.index]
lab = pd.concat([lab, fb.loc[miss]]) if miss else lab
G = torch.tensor(lab.loc[ids].values.astype(np.float32), device=dev)
tr = pd.read_csv(os.path.join(COMP, "train.csv"))
gold = tr[tr["ACL"].notna()].set_index("StudyInstanceUID")
gm = np.array([u in set(gold.index) for u in ids])
val, train = np.where(gm)[0], np.where(~gm)[0]
Yg = gold.loc[ids[val], LABELS].values.astype(int)
Et = torch.from_numpy(np.asarray(E)).to(dev)              # [n, 36, D] fp16


class AttnHead(nn.Module):
    """Identical to every other member's head."""

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


def gold_auc(p):
    return float(np.mean([roc_auc_score(Yg[:, j], p[:, j]) for j in range(len(LABELS))
                          if len(np.unique(Yg[:, j])) == 2]))


torch.manual_seed(0)
D = Et.shape[-1]
head = AttnHead(D, len(LABELS)).to(dev)
EPOCHS, BS = 40, 32
opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-2)
steps = EPOCHS * (len(train) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps)
bce = nn.BCEWithLogitsLoss()
best, best_p, best_ep = -1, None, -1
print("\nepoch  loss    gold", flush=True)
for ep in range(EPOCHS):
    head.train()
    perm = torch.from_numpy(np.random.default_rng(ep).permutation(train)).to(dev)
    tot = 0.0
    for b in range(len(train) // BS):
        idx = perm[b * BS:(b + 1) * BS]
        loss = bce(head(Et[idx].float()), G[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
    head.eval()
    with torch.no_grad():
        p = head(Et[torch.from_numpy(val).to(dev)].float()).cpu().numpy()
    g = gold_auc(p)
    if (ep + 1) % 4 == 0:
        print(f"{ep+1:>5} {tot/(len(train)//BS):.4f} {g:.4f}", flush=True)
    if g > best:
        best, best_p, best_ep = g, p.copy(), ep + 1
        torch.save({"head": head.state_dict(), "labels": LABELS, "dim": D,
                    "size_in": SIZE_IN, "arch": "medimageinsight_davit_frozen",
                    "gold": g}, "/kaggle/working/mii_head.pt")

print(f"\nbest gold {best:.4f} at epoch {best_ep}", flush=True)
print("  bar: competitive means within ~0.01 of v19B 0.8700 (best single member);")
print("  frozen DINOv2 + head was far below fine-tuned members on this project.")
np.savez_compressed("/kaggle/working/mii_gold_preds.npz",
                    study_ids=ids[val], pred=1 / (1 + np.exp(-best_p)))
print("DONE", flush=True)
