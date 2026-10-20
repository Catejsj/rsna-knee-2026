"""Can we get CUDA working on Kaggle's P100 by installing an older torch?

This is the constraint that has shaped every decision so far. Kaggle assigns a
Tesla P100 (compute capability sm_60) while its preinstalled PyTorch is built
for sm_70+, so every CUDA call raises `no kernel image is available` and
everything silently falls back to CPU at ~14 slices/s. That is why we run a
frozen encoder with a linear head instead of fine-tuning like the 0.809 public
baseline does.

But PyTorch wheels built against **CUDA 11.8** still carry sm_60 kernels. Kernels
with internet enabled can install one. If that works:

  * encoding goes from ~5 s/study to well under 0.1
  * **fine-tuning the last encoder blocks becomes possible**, which is the
    single structural difference between our pipeline and the ones above 0.8

Inference does not need this — the submission notebook has no internet, and CPU
inference already fits the 9 h budget. Only training needs the GPU.

Measures three things: does CUDA initialise, is a forward pass correct, and how
fast is forward+backward on DINOv2 (the thing fine-tuning needs).
"""
import subprocess
import sys
import time

print("=== preinstalled torch ===", flush=True)
import torch
print(f"torch {torch.__version__}  cuda={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"capability: {torch.cuda.get_device_capability(0)}")
    try:
        torch.zeros(8, device="cuda").mul_(2).cpu()
        print("preinstalled torch: CUDA WORKS (nothing to fix)")
    except Exception as e:
        print(f"preinstalled torch: CUDA BROKEN -> {str(e)[:90]}")

print("\n=== installing torch 2.6.0+cu118 (sm_60 kernels AND satisfies transformers>=2.6) ===",
      flush=True)
t0 = time.time()
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch==2.6.0", "torchvision==0.21.0",
     "--index-url", "https://download.pytorch.org/whl/cu118"],
    capture_output=True, text=True)
print(f"pip rc={r.returncode} in {time.time()-t0:.0f}s")
if r.returncode != 0:
    print((r.stdout + r.stderr)[-1500:])
    sys.exit("install failed")

# torch must be re-imported in a fresh process to pick up the new build
probe = r'''
import time, torch
print(f"torch {torch.__version__}  cuda={torch.version.cuda}", flush=True)
print(f"available={torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("no cuda")
print(f"device: {torch.cuda.get_device_name(0)} "
      f"capability {torch.cuda.get_device_capability(0)}", flush=True)
x = torch.randn(1024, 1024, device="cuda")
y = (x @ x).sum().item()
print(f"matmul on GPU OK (checksum {y:.1f})", flush=True)

from transformers import AutoModel
import glob, os
root = [os.path.dirname(p) for p in
        glob.glob("/kaggle/input/**/config.json", recursive=True)
        if "dinov2" in p.lower()][0]
try:
    m = AutoModel.from_pretrained(root).cuda()
except Exception as ex:
    print("from_pretrained failed:", str(ex)[:120], flush=True)
    from transformers import AutoConfig, Dinov2Model
    import safetensors.torch as st, glob as g
    cfg = AutoConfig.from_pretrained(root)
    m = Dinov2Model(cfg)
    sf = g.glob(os.path.join(root, "*.safetensors"))
    if sf:
        sd = st.load_file(sf[0])
    else:
        sd = torch.load(g.glob(os.path.join(root, "*.bin"))[0],
                        map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(
        {k.replace("dinov2.", ""): v for k, v in sd.items()}, strict=False)
    print(f"manual load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    m = m.cuda()

# inference speed at both resolutions
m.eval()
for px in (224, 336):
    b = torch.randn(24, 3, px, px, device="cuda")
    with torch.no_grad():
        for _ in range(2): m(pixel_values=b)
        torch.cuda.synchronize(); t=time.time()
        for _ in range(5): m(pixel_values=b)
        torch.cuda.synchronize()
    per = (time.time()-t)/5
    print(f"INFERENCE {px}px: {24/per:7.1f} slices/s "
          f"(CPU measured 14) -> 4407 studies x 24 slices = "
          f"{4407*24/(24/per)/3600:.2f} h", flush=True)

# fine-tuning speed: forward + backward with the last 4 blocks trainable
m.train()
for p in m.parameters(): p.requires_grad = False
for blk in m.encoder.layer[-4:]:
    for p in blk.parameters(): p.requires_grad = True
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
b = torch.randn(8, 3, 224, 224, device="cuda")
torch.cuda.synchronize(); t=time.time()
for _ in range(5):
    out = m(pixel_values=b).last_hidden_state[:,0].mean()
    out.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()
per = (time.time()-t)/5
print(f"FINE-TUNE fwd+bwd: {8/per:.1f} img/s at 224px, batch 8", flush=True)
print(f"  -> one epoch over 4407 studies x 24 slices = "
      f"{4407*24/(8/per)/3600:.2f} h", flush=True)
print(f"  -> 10 epochs = {10*4407*24/(8/per)/3600:.1f} h", flush=True)
print("VERDICT: GPU usable, fine-tuning is on the table", flush=True)
'''
print("\n=== re-testing in a fresh process ===", flush=True)
r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr[-1200:])
print("DONE", flush=True)
