"""4-model ensemble with the improved RadImageNet members. Internet OFF, CPU.

    v10   DINOv2-small,  LLM labels          gold 0.8438  corr 0.907
    v11   DINOv2-base,   LLM labels          gold 0.8541  corr 0.911
    v17A  RadImageNet, leehans, layer4       gold 0.8585  corr 0.920
    v17B  RadImageNet, pilkwang, layer3+4    gold 0.8654  corr 0.926
    -----------------------------------------------------------------
    averaged                                 gold 0.8846   (shipped: 0.8781)

**+0.0065 at identical runtime.** Same ~7.5 h as the ensemble that scored
LB 0.900, because the two new members are ResNets (~1 h each) replacing three
older ones.

The idea behind them: **training effort is free at submission time.** A
ResNet-50 costs ~1 h of the 9 h cap however it was trained, so the old cautious
settings (layer4 only, pilkwang teacher, gold 0.8508) were leaving score on the
table for nothing. Retraining the same 1 h slot harder gives 0.8585 (better
teacher) and 0.8654 (layer3+4 unfrozen) -- and **v17B is now the best single
model of the project, beating DINOv2-base at a third of its inference cost.**

That last point retracts a rule this project had held since the beginning:
"add information, not parameters". Unfreezing layer3 is pure added capacity and
it won by the larger of the two margins tested. Every earlier capacity test
predates the LLM teachers; under a good teacher the old rule does not hold.

Why this was the only lever left, all measured:
  * slice reduction 12->8 per plane costs -0.0128 gold for 0.67x time -- net loss
  * six cheap members plateau at 0.8739; they correlate with each other
  * base earns its 3 h (best base-free at equal cost is 0.8739 vs 0.8781)

Chosen under the 9 h cap, not from the global ranking. The best combination
overall is base+leehans+336+radA+radB at 0.8853, but the 336px member needs its
own decode pass and the total lands ~11 h. This one is built to the exact cost
of a submission proven to complete.
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

MEMBERS = ["finetuned_llm.pt", "finetuned_llm_base.pt",
           "finetuned_radimg_A.pt", "finetuned_radimg_B.pt"]
PATHS = []
for _n in MEMBERS:
    _f = glob.glob(f"/kaggle/input/**/{_n}", recursive=True)
    assert _f, f"missing checkpoint {_n}"
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
