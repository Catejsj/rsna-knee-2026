"""Robustness hedge: five members, no DINOv2-base. Internet OFF, CPU, ~6.5 h.

    v10   DINOv2-small, pilkwang LLM labels       gold 0.8438
    v12   DINOv2-small, leehans LLM labels        gold 0.8474
    v17A  RadImageNet R50, leehans, layer4        gold 0.8585
    v17B  RadImageNet R50, pilkwang, layer3+4     gold 0.8653
    v19B  RadImageNet R50, layer3+4, 4349 studies gold 0.8700
    -----------------------------------------------------------------
    averaged                                      gold 0.8838   (ens7: 0.8861)

Found by `src/ensemble_cost_search.py`: the best combination that does not use
DINOv2-base. Base costs 3 of the 9 hours on its own and buys only +0.0022 gold,
inside the noise floor (same-recipe reruns differ by ~0.008).

**This is not trying to beat ens7.** It is the second of the two final
selections. A timeout scores zero, not a lower number -- that already happened
once on this project. ens7 runs ~7.5 h; this runs ~6.5 h with no base, so it has
an extra hour of headroom and a structurally different member set.
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

# gold recorded inside each checkpoint at training time. Asserting on it is the
# only reliable guard against a stale dataset version: same filename, same
# tensor shapes, same byte count -- only the recorded metric differs.
MEMBERS = {"finetuned_llm.pt": 0.8438, "finetuned_leehans.pt": 0.8474,
           "finetuned_radimg_A.pt": 0.8585, "finetuned_radimg_B.pt": 0.8653,
           "finetuned_full_B.pt": 0.8700}
PATHS = []
for _n, _g in MEMBERS.items():
    _f = glob.glob(f"/kaggle/input/**/{_n}", recursive=True)
    assert _f, f"missing checkpoint {_n}"
    _rec = float(torch.load(_f[0], map_location="cpu", weights_only=False)["gold"])
    assert abs(_rec - _g) < 5e-4, f"{_n}: recorded gold {_rec:.4f}, expected {_g} -- stale?"
    PATHS.append(_f[0])
CKPT = PATHS[0]
COMP = os.path.dirname(glob.glob("/kaggle/input/**/test_series.csv", recursive=True)[0])
print(f"checkpoint: {CKPT}\ncompetition: {COMP}", flush=True)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
LABELS = list(ck["labels"])
SIZE, SLICES = int(ck["size"]), int(ck["slices"])
PLANES = tuple(ck["planes"])
print(f"model: {SIZE}px x {SLICES} slices, planes={'+'.join(PLANES)}, "
      f"val gold {ck.get('gold', float('nan')):.4f}", flush=True)

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
import json as _json
_roots = {}
for _p in glob.glob("/kaggle/input/**/config.json", recursive=True):
    if "dinov2" not in _p.lower():
        continue
    try:
        _h = _json.load(open(_p)).get("hidden_size")
    except Exception:
        continue
    if _h:
        _roots[int(_h)] = os.path.dirname(_p)
# both small (384) and base (768) are attached, so pick by the checkpoint's own
# head LayerNorm rather than by whichever folder glob happens to return first
_dim = int(ck["head"]["norm.weight"].shape[0])
assert _dim in _roots, f"no encoder with hidden_size {_dim}; have {list(_roots)}"
root = _roots[_dim]
print(f"encoder hidden_size {_dim} -> {root}", flush=True)
import torchvision


class ResNetTrunk(torch.nn.Module):
    """ResNet-50 to layer4 + global average pool, so it exposes the same
    [B, dim] interface the DINOv2 members do and predict_study needs no
    per-architecture branching."""

    def __init__(self, seq):
        super().__init__()
        self.seq = seq

    def forward(self, pixel_values):
        return self.seq(pixel_values).mean(dim=(2, 3))


MODELS = []
for _path in PATHS:
    _c = torch.load(_path, map_location="cpu", weights_only=False)
    _d = int(_c["head"]["norm.weight"].shape[0])
    if _c.get("arch") == "resnet50_radimagenet":
        _net = torchvision.models.resnet50(weights=None)
        _b = ResNetTrunk(torch.nn.Sequential(*list(_net.children())[:-2]))
        _b.seq.load_state_dict(_c["backbone"], strict=True)
        _kind = "resnet50-radimagenet"
    else:
        assert _d in _roots, f"no encoder with hidden_size {_d}"
        _b = AutoModel.from_pretrained(_roots[_d])
        _b.load_state_dict(_c["backbone"])
        _kind = f"dinov2-{_d}"
    _h = AttnHead(_d, len(LABELS))
    _h.load_state_dict(_c["head"])
    _b.eval(); _h.eval()
    MODELS.append((_b, _h))
    print(f"  {os.path.basename(_path)}: {_kind}, dim {_d}, val gold "
          f"{_c.get('gold', float('nan')):.4f}", flush=True)
backbone, head = MODELS[0]
torch.set_num_threads(os.cpu_count() or 4)
print(f"weights loaded, {os.cpu_count()} CPU threads", flush=True)


@torch.no_grad()
def predict_study(stacks):
    """list of [k,H,W] float arrays (one per plane) -> 12 probabilities."""
    imgs = np.concatenate(stacks, axis=0)
    x = torch.from_numpy(imgs).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    probs = np.zeros(len(LABELS), dtype=np.float64)
    for _b, _h in MODELS:
        emb = []
        for i in range(0, len(x), 12):
            o = _b(pixel_values=x[i:i + 12])
            emb.append(o if torch.is_tensor(o) else o.last_hidden_state[:, 0])
        probs += torch.sigmoid(_h(torch.cat(emb).unsqueeze(0))[0]).numpy()
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
