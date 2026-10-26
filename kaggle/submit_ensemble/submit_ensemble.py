"""Submission for the 3-model prediction ensemble. Internet OFF, CPU.

Members: v7 (gold 0.7353) + batch configs A_e4_u4 (0.7273) and C_e4_u6
(0.7309). Averaged, they score **gold 0.7396**.

The previous session searched only subsets of {A,B,C,D}, found A+C at 0.7347 --
just under v7 -- and concluded prediction-averaging does not help. That
conclusion was an artifact of the pool: v7 was never in it, because the batch
run produced no v7 validation predictions to average. Adding the strongest
member turns -0.0006 into +0.0043. Ensembling was fine; the search was
incomplete.

+0.0043 is well inside gold-58's +/-0.05 noise, which is exactly why this is
worth a submission: the leaderboard has ~1300 studies against gold's 58 and can
resolve what gold cannot.

**TTA was measured at the same time and is a settled negative** -- hflip made
every single model worse (v7 0.7353 -> 0.7305) and lost in all 31 combinations.
Geometry normalization already flips every knee to the same side, so a test-time
flip presents the encoder with mirrored anatomy it never trained on. Not shipped,
and no submission was spent finding that out.

Runtime: 3 members x ~1 h = ~3 h on CPU, inside the 12 h no-GPU budget.

--- original single-model header follows ---

Submission for the fine-tuned model. Internet OFF, CPU.

Fine-tuning lifted gold-58 from 0.6811 (frozen encoder + ridge, LB 0.742) to
0.7353: the largest measured gain of the project. This ships it.

Runs on CPU deliberately. Training needed `torch==2.6.0+cu118` to drive the
P100, but that install needs internet, which submission kernels do not have.
CPU inference at ~14 slices/s over 1300 studies x 36 slices is ~0.93 h, well
inside the 9 h budget, so there is no reason to take on the risk of shipping a
CUDA wheel as a dataset.

Everything about the representation (size, slice count, planes) is read from
the checkpoint rather than restated, so this cannot disagree with what the
weights were trained on. Geometry normalization comes from the same shared
`rsna_features` module the cache was built with.
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys

sys.path.insert(0, os.path.dirname(
    glob.glob("/kaggle/input/**/rsna_features.py", recursive=True)[0]))
import rsna_features as F

# v7 + the two batch configs that survived the search. Averaging their
# *predictions* (not concatenating their features, which is what every earlier
# "combination" experiment did) scored gold 0.7396 against v7's 0.7353.
MEMBERS = ["finetuned.pt", "model_A_e4_u4.pt", "model_C_e4_u6.pt"]
PATHS = []
for name in MEMBERS:
    found = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert found, f"missing checkpoint {name}"
    PATHS.append(found[0])
COMP = os.path.dirname(glob.glob("/kaggle/input/**/test_series.csv", recursive=True)[0])
print("checkpoints:", flush=True)
for p in PATHS:
    print("  ", p, flush=True)
print(f"competition: {COMP}", flush=True)

ck = torch.load(PATHS[0], map_location="cpu", weights_only=False)
LABELS = list(ck["labels"])
SIZE, SLICES = int(ck["size"]), int(ck["slices"])
PLANES = tuple(ck["planes"])
print(f"model: {SIZE}px x {SLICES} slices, planes={'+'.join(PLANES)}, "
      f"{len(PATHS)} members", flush=True)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class AttnHead(nn.Module):
    """Must match the training definition exactly."""

    def __init__(self, dim, n_out, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(n_out, dim) * dim ** -0.5)
        self.value = nn.Linear(dim, hidden)
        self.out = nn.Parameter(torch.zeros(n_out, hidden))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, emb):
        e = self.norm(emb)
        attn = torch.einsum("bsd,cd->bcs", e, self.query) * e.shape[-1] ** -0.5
        attn = attn.softmax(-1)
        pooled = torch.einsum("bcs,bsh->bch", attn, self.value(e))
        return (pooled * self.out).sum(-1) + self.bias


from transformers import AutoModel
root = [os.path.dirname(p) for p in
        glob.glob("/kaggle/input/**/config.json", recursive=True)
        if "dinov2" in p.lower()][0]
MODELS = []
for path in PATHS:
    c = torch.load(path, map_location="cpu", weights_only=False)
    b = AutoModel.from_pretrained(root)
    b.load_state_dict(c["backbone"])
    h = AttnHead(b.config.hidden_size, len(LABELS))
    h.load_state_dict(c["head"])
    b.eval(); h.eval()
    MODELS.append((b, h))
    print(f"  loaded {os.path.basename(path)} (val gold "
          f"{c.get('gold', float('nan')):.4f})", flush=True)
torch.set_num_threads(os.cpu_count() or 4)
print(f"{len(MODELS)} models loaded, {os.cpu_count()} CPU threads", flush=True)


@torch.no_grad()
def predict_study(stacks):
    """list of [k,H,W] float arrays (one per plane) -> 12 probabilities.

    Each member sees identical pixels; only the weights differ. Averaging the
    sigmoids rather than the logits is what was scored on gold, so it is what
    ships -- the two are not the same ranking once members disagree.
    """
    imgs = np.concatenate(stacks, axis=0)
    x = torch.from_numpy(imgs).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    probs = np.zeros(len(LABELS), dtype=np.float64)
    for backbone, head in MODELS:
        emb = []
        for i in range(0, len(x), 12):
            emb.append(backbone(pixel_values=x[i:i + 12]).last_hidden_state[:, 0])
        logits = head(torch.cat(emb).unsqueeze(0))[0]
        probs += torch.sigmoid(logits).numpy()
    return (probs / len(MODELS)).astype(np.float32)


test = pd.read_csv(os.path.join(COMP, "test.csv"))
ser = pd.read_csv(os.path.join(COMP, "test_series.csv"))
root_dir = os.path.join(COMP, "test_series")
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
print(f"test studies: {len(test)}", flush=True)

# fallback if a study yields no usable pixels at all
PRIOR = np.full(len(LABELS), 0.5, dtype=np.float32)

rows, n_fallback = [], 0
t0 = time.time()
for i, uid in enumerate(test.StudyInstanceUID):
    stacks = []
    if uid in by_study:
        for plane in PLANES:
            row = F.pick_series_for_plane(by_study[uid], plane)
            stack = None
            if row is not None:
                slices = F.load_series_with_geometry(
                    os.path.join(root_dir, uid, row.SeriesInstanceUID),
                    max_slices=32)
                if len(slices) >= 3:
                    stack = F.geometric_slice_stack(slices, n_slices=SLICES,
                                                    size=SIZE)
            # a missing plane becomes zeros, matching how the training cache
            # was built (that block was simply never written)
            stacks.append(stack if stack is not None
                          else np.zeros((SLICES, SIZE, SIZE), dtype=np.float32))
            if stack is not None and len(stack) < SLICES:
                pad = np.zeros((SLICES, SIZE, SIZE), dtype=np.float32)
                pad[:len(stack)] = stack
                stacks[-1] = pad
    if not stacks or all(s.max() == 0 for s in stacks):
        probs = PRIOR
        n_fallback += 1
    else:
        probs = predict_study(stacks)
    rows.append([uid] + [float(v) for v in probs])
    if (i + 1) % 50 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(test)} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(len(test)-i-1)/60:.0f} min", flush=True)

el = time.time() - t0
sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"\nwrote submission.csv {sub.shape}", flush=True)
print(f"fallback-to-prior: {n_fallback}/{len(test)}", flush=True)
print(f"total {el/3600:.2f} h of the 9 h budget", flush=True)
print(sub.head().to_string(), flush=True)
print("DONE", flush=True)
