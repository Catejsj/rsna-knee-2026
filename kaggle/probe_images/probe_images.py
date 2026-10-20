"""Does unsupervised grouping of knee MRI find DISEASE, or ACQUISITION?

Proposal under test: skip the text, group the images, label the groups.
The prediction being tested is that image similarity is dominated by scanner,
sequence and plane -- which change every pixel -- rather than by pathology,
which changes a fraction of a percent of them.

Two analyses, both measured with adjusted mutual information (AMI: 0 = the
grouping tells you nothing about the target, 1 = it tells you everything) plus
a supervised linear probe, which separates "the signal is absent from these
features" from "the signal is present but clustering does not surface it".

  A. series level, all planes   -> AMI(clusters, plane / fluid-sensitive)
  B. study level, sagittal FS   -> AMI(clusters, each of the 12 findings)

If A is high and B is near zero, grouping images cannot produce labels.
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

N_SERIES_A = 500      # series for the acquisition test
N_STUDIES_B = 300     # studies for the pathology test
SLICES_B = 5          # slices averaged per study
SIZE = 64

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def find_input():
    """Recursive: attached sources mount at /kaggle/input/competitions/<slug>/
    and /kaggle/input/datasets/<user>/<slug>/, not the flat /kaggle/input/<slug>/."""
    comp = glob.glob("/kaggle/input/**/train_series.csv", recursive=True)
    lab = glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)
    assert comp, "competition data not attached"
    return os.path.dirname(comp[0]), (os.path.dirname(lab[0]) if lab else None)


COMP, DERIVED = find_input()
print("competition:", COMP, "\nderived:", DERIVED, flush=True)


# ------------------------------------------------------------------ dicom io
import pydicom


def load_slice(path):
    """Return a normalized SIZE x SIZE float array, or None if unreadable."""
    try:
        ds = pydicom.dcmread(path)
        a = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    if a.ndim != 2 or min(a.shape) < 16:
        return None
    # per-slice intensity normalization: MRI has no absolute scale
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return None
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    # cheap resize by strided averaging
    h, w = a.shape
    yi = (np.arange(SIZE) * h / SIZE).astype(int)
    xi = (np.arange(SIZE) * w / SIZE).astype(int)
    return a[np.ix_(yi, xi)]


def series_files(study, series):
    d = os.path.join(COMP, "train_series", study, series)
    try:
        return sorted(os.listdir(d)), d
    except Exception:
        return [], d


# ------------------------------------------------------------------- sampling
ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
print(f"train_series rows: {len(ser)}", flush=True)

# --- A: series stratified across planes
a_rows = (ser.groupby("Anatomical_Plane", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), N_SERIES_A // 3), random_state=0)))
print(f"A: sampling {len(a_rows)} series", flush=True)

A_feat, A_plane, A_fluid = [], [], []
for i, r in enumerate(a_rows.itertuples()):
    files, d = series_files(r.StudyInstanceUID, r.SeriesInstanceUID)
    if not files:
        continue
    img = load_slice(os.path.join(d, files[len(files) // 2]))
    if img is None:
        continue
    A_feat.append(img.ravel())
    A_plane.append(r.Anatomical_Plane)
    A_fluid.append(r.Fluid_Sensitive)
    if (i + 1) % 100 == 0:
        print(f"  A {i+1}/{len(a_rows)} loaded={len(A_feat)}", flush=True)
A_feat = np.array(A_feat)
print(f"A: {A_feat.shape}", flush=True)

# --- B: studies, sagittal fluid-sensitive series only (what we would train on)
sag = ser[(ser.Anatomical_Plane == "Sagittal") & (ser.Fluid_Sensitive == 1)]
sag = sag.drop_duplicates("StudyInstanceUID")
sag = sag.sample(min(len(sag), N_STUDIES_B), random_state=0)
print(f"B: sampling {len(sag)} studies", flush=True)

B_feat, B_study = [], []
for i, r in enumerate(sag.itertuples()):
    files, d = series_files(r.StudyInstanceUID, r.SeriesInstanceUID)
    if len(files) < SLICES_B:
        continue
    idx = np.linspace(0.2, 0.8, SLICES_B) * (len(files) - 1)
    imgs = [load_slice(os.path.join(d, files[int(k)])) for k in idx]
    imgs = [x for x in imgs if x is not None]
    if len(imgs) < 3:
        continue
    B_feat.append(np.mean(imgs, axis=0).ravel())
    B_study.append(r.StudyInstanceUID)
    if (i + 1) % 50 == 0:
        print(f"  B {i+1}/{len(sag)} loaded={len(B_feat)}", flush=True)
B_feat = np.array(B_feat)
print(f"B: {B_feat.shape}", flush=True)


# --------------------------------------------------------------- deep features
def deep_features(raw):
    """ImageNet ResNet18 penultimate features -- a fairer stand-in for 'good
    image features' than raw pixels. Falls back to raw if weights unavailable."""
    try:
        import torch
        import torchvision
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        m.fc = torch.nn.Identity()
        m.eval()
        # Kaggle's GPU + preinstalled torch can mismatch on CUDA arch
        # ("no kernel image is available"), so verify with a real op first.
        dev = "cpu"
        if torch.cuda.is_available():
            try:
                torch.zeros(8, device="cuda").mul_(2).cpu()
                dev = "cuda"
            except Exception as e:
                print("cuda unusable, falling back to cpu:", str(e)[:80], flush=True)
        m = m.to(dev)
        x = torch.tensor(raw.reshape(-1, 1, SIZE, SIZE), dtype=torch.float32)
        x = x.repeat(1, 3, 1, 1)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                out.append(m(x[i:i + 64].to(dev)).cpu().numpy())
        return np.vstack(out)
    except Exception as e:
        print("deep features unavailable:", e, flush=True)
        return None


# ------------------------------------------------------------------- analysis
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


def prep(X, n=50):
    X = StandardScaler().fit_transform(X)
    return PCA(n_components=min(n, X.shape[0] - 1, X.shape[1])).fit_transform(X)


def probe_auc(X, y):
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < 10:
        return float("nan")
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=4,
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def report(title, X, targets, ks=(3, 8, 20)):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    Z = prep(X)
    print(f"{'target':<22}" + "".join(f"{'AMI k='+str(k):>12}" for k in ks)
          + f"{'probe AUC':>12}")
    clusterings = {k: KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
                   for k in ks}
    for name, y in targets.items():
        cells = "".join(f"{ami(y, clusterings[k]):>12.3f}" for k in ks)
        yb = np.asarray(y)
        auc = probe_auc(Z, yb) if len(np.unique(yb)) == 2 else float("nan")
        print(f"{name:<22}{cells}{auc:>12.3f}")


report("A. SERIES LEVEL -- can grouping recover the acquisition settings?",
       A_feat,
       {"Anatomical_Plane": np.array(A_plane),
        "Fluid_Sensitive": np.array(A_fluid, dtype=int)})

if DERIVED:
    lab = pd.read_csv(os.path.join(DERIVED, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
    tgt = {}
    for l in LABELS:
        y = (lab.loc[B_study, l].values >= 0.5).astype(int)
        if min(np.bincount(y, minlength=2)) >= 15:
            tgt[l] = y
    report("B. STUDY LEVEL -- can grouping recover the 12 findings?", B_feat, tgt)

D = deep_features(B_feat)
if D is not None and DERIVED:
    report("B2. STUDY LEVEL, ResNet18 features instead of raw pixels", D, tgt)
DA = deep_features(A_feat)
if DA is not None:
    report("A2. SERIES LEVEL, ResNet18 features", DA,
           {"Anatomical_Plane": np.array(A_plane),
            "Fluid_Sensitive": np.array(A_fluid, dtype=int)})

print("\nDONE", flush=True)
