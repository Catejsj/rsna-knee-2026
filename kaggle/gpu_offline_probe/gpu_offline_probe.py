"""Does an internet-OFF GPU kernel (the submission environment) give a usable GPU
with the STOCK torch? If it is a T4 (sm_75), yes. If a P100 (sm_60), the stock
build raises 'no kernel image' and we need an offline cu118 wheel."""
import time, torch, torchvision
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0),
          "| arch list:", torch.cuda.get_arch_list(), flush=True)
    try:
        m = torchvision.models.resnet50(weights=None).cuda().eval()
        x = torch.randn(36, 3, 224, 224, device="cuda")
        with torch.no_grad():
            m(x); torch.cuda.synchronize(); t = time.time()
            for _ in range(5): m(x)
            torch.cuda.synchronize()
        print(f"RESNET50 OK on GPU: {5*36/(time.time()-t):.0f} slices/s", flush=True)
    except Exception as e:
        print("GPU FORWARD FAILED:", str(e)[:200], flush=True)
print("DONE", flush=True)
