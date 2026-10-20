"""Is Laya a better teacher? Gate test on gold-58 only. CPU, ~15 min.

Laya (`convaiinnovations/laya-multilingual`, Apache-2.0, mmBERT-base, 322M) is
the open-weight counterpart of TypeSafe's Jev: it answers typed questions --
`score` (ordered levels) and `noul` (yes/no probability) -- in one forward pass,
with calibrated probabilities, in 100+ languages.

**Why not Jev itself.** Jev is proprietary and API-only. The competition's
data-use terms forbid sending report text to an external LLM API, so every
teacher on this project has run inside Kaggle. Laya runs locally: its core only
contacts Hugging Face to download weights. Outbound calls exist solely in the
LangChain/CrewAI integrations when given a remote `base_url`, which we never use.

**Why gold-58 only.** The label lever has saturated: the leehans teacher beat
pilkwang by +0.023 on gold and the model trained on it gained only +0.004. So a
new teacher has to clear a high bar before labelling all 4407 reports and
retraining is worth anything. This scores Laya against both public teachers on
the SAME 58 studies and stops there.

    bar to clear: leehans v4_blend 0.8932  (pilkwang v2 0.8700)

**The questions were written without reading any report.** That is the honest
role for a large model here: specify what each label means (compartment, side,
what counts as present) -- which is what made the rules extractor work -- not
read and label the competition data itself.

Two arms, asked in the same forward pass:
    score   5 ordered levels, from "explicitly normal" to "moderate or severe";
            the expected level is a graded value like our training targets
    noul    calibrated probability that the finding is present
"""
import glob
import os
import subprocess
import sys
import time

t0 = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.3.24"],
                   capture_output=True, text=True)
print(f"pip laya rc={r.returncode} in {time.time()-t0:.0f}s", flush=True)
if r.returncode != 0:
    print((r.stdout + r.stderr)[-1500:])
    sys.exit("install failed")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

# What each label means, stated precisely. Side and compartment are spelled out
# because gating on them was the single biggest source of extractor error
# (medial-vs-lateral meniscus, the Turkish "medyal" miss).
DEFS = {
    "ACL": "a tear of the anterior cruciate ligament (ACL), partial or complete",
    "MCL": "an injury, sprain or tear of the medial collateral ligament (MCL), any grade",
    "Medial Meniscus": "a tear of the MEDIAL meniscus (not the lateral meniscus)",
    "Lateral Meniscus": "a tear of the LATERAL meniscus (not the medial meniscus)",
    "Medial OA": "osteoarthritis, cartilage loss or chondral damage in the MEDIAL "
                 "tibiofemoral compartment (medial femoral condyle or medial tibial plateau)",
    "Lateral OA": "osteoarthritis, cartilage loss or chondral damage in the LATERAL "
                  "tibiofemoral compartment (lateral femoral condyle or lateral tibial plateau)",
    "PF OA": "osteoarthritis, chondromalacia, cartilage loss or chondral damage in the "
             "PATELLOFEMORAL compartment (patella or trochlea)",
    "Effusion": "a knee joint effusion (excess fluid in the joint)",
    "Synovitis": "synovitis, synovial thickening or synovial proliferation",
    "Baker's": "a popliteal (Baker's) cyst",
    "Contusion": "a bone contusion, bone bruise or bone marrow edema",
    "Fracture": "a fracture of any bone, including osteochondral, avulsion or impaction fractures",
}
LEVELS = ["explicitly described as normal, intact or absent",
          "not mentioned",
          "possible, suspected or equivocal",
          "present, mild",
          "present, moderate or severe"]

# Two 12-question sets in two calls rather than 24 in one: the question head has
# its own token budget, and overrunning it degrades answers without an error.
QS, QN = {}, {}
for i, lab in enumerate(LABELS):
    QS[f"s{i}"] = {"type": "score",
                  "instructions": f"In this knee MRI report, what does the report say about "
                                  f"{DEFS[lab]}?",
                  "criteria": LEVELS}
    QN[f"n{i}"] = {"type": "noul",
                  "instructions": f"Does this knee MRI report state that there is {DEFS[lab]}?"}

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
train = pd.read_csv(os.path.join(COMP, "train.csv"))
gold = train[train["ACL"].notna()].reset_index(drop=True)
Y = gold[LABELS].values.astype(int)
print(f"gold studies: {len(gold)} | report chars median "
      f"{int(gold.Report.str.len().median())}", flush=True)

from laya import Router
router = Router()

# ---- throughput probe (local estimates have been 3.5x optimistic 4 times) ----
def ask(text, qs):
    return router.predict(text, qs, model="multilingual", max_len=2048)


_t = time.time()
ask(gold.Report.iloc[0], QS); ask(gold.Report.iloc[0], QN)
first = time.time() - _t                            # includes weight download
_t = time.time()
for k in range(1, 4):
    ask(gold.Report.iloc[k], QS); ask(gold.Report.iloc[k], QN)
per = (time.time() - _t) / 3
print(f"PROBE: first call {first:.0f}s (download+load), then {per:.2f} s/report "
      f"-> gold-58 {per*58/60:.1f} min, all 4407 would be {per*4407/3600:.1f} h",
      flush=True)

S = np.zeros((len(gold), len(LABELS)))
N = np.zeros((len(gold), len(LABELS)))
routed = {}
for r_i, rep in enumerate(gold.Report.fillna("")):
    out = ask(rep, QS)
    a = dict(out["answers"])
    a.update(ask(rep, QN)["answers"])
    for j in range(len(LABELS)):
        S[r_i, j] = a[f"s{j}"]["score"]
        N[r_i, j] = a[f"n{j}"]["noul"]
    m = out.get("routing", {}).get("model", "?")
    routed[m] = routed.get(m, 0) + 1
print(f"routed to: {routed}", flush=True)


# Every arm is scored on the SAME studies. leehans covers 57 of the 58 gold
# studies, so comparing it on 57 against Laya on 58 would not be like for like.
KEEP = np.ones(len(gold), dtype=bool)


def per_label(P):
    out = {}
    for j, lab in enumerate(LABELS):
        y = Y[KEEP, j]
        if len(np.unique(y)) == 2:
            out[lab] = roc_auc_score(y, P[KEEP, j])
    return out


def teacher(pattern):
    f = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not f:
        print(f"  {pattern}: not attached", flush=True)
        return None
    t = pd.read_csv(f[0]).set_index("StudyInstanceUID")
    have = np.array([u in t.index for u in gold.StudyInstanceUID])
    if not have.all():
        print(f"  {pattern}: covers {have.sum()}/{len(gold)} gold studies", flush=True)
    T = np.full((len(gold), len(LABELS)), np.nan)
    T[have] = t.loc[gold.StudyInstanceUID[have], LABELS].values.astype(float)
    return T, have


arms = {"laya_score": S, "laya_noul": N}
for name, pat in [("pilkwang", "report_labels_v2.csv"),
                  ("leehans", "llm_labels_v4_blend.csv"),
                  ("rules_v5", "labels_rules_v5.csv")]:
    got = teacher(pat)
    if got is not None:
        arms[name], have = got
        KEEP &= have
print(f"scoring every arm on the {KEEP.sum()} studies all sources cover", flush=True)

tab = {k: per_label(v) for k, v in arms.items()}
print("\nper-label AUC on gold-58", flush=True)
print(f"  {'label':<18}" + "".join(f"{k:>12}" for k in tab), flush=True)
for lab in LABELS:
    if lab in tab["laya_score"]:
        print(f"  {lab:<18}" + "".join(f"{tab[k][lab]:>12.4f}" for k in tab), flush=True)
print(f"  {'MACRO':<18}" + "".join(f"{np.mean(list(tab[k].values())):>12.4f}"
                                     for k in tab), flush=True)

# Decorrelation from the best teacher matters for the same reason it does for
# ensemble members: a mediocre teacher that is wrong in DIFFERENT places can
# still improve a blend. Reported, not acted on -- blends fitted on 58 studies
# are how this project has fooled itself before (6-way blend: +0.0036, noise).
if "leehans" in arms:
    from scipy.stats import rankdata
    def rk(P):
        return np.apply_along_axis(rankdata, 0, P)
    for k in ("laya_score", "laya_noul"):
        c = np.mean([np.corrcoef(arms[k][KEEP, j], arms["leehans"][KEEP, j])[0, 1]
                     for j in range(len(LABELS))])
        blend = np.zeros_like(arms[k])
        blend[KEEP] = (rk(arms[k][KEEP]) + rk(arms["leehans"][KEEP])) / 2
        print(f"\n  {k}: per-label corr with leehans {c:.3f} | rank-blend macro "
              f"{np.mean(list(per_label(blend).values())):.4f} (descriptive only)",
              flush=True)

np.savez_compressed("/kaggle/working/laya_gold.npz",
                    study_ids=gold.StudyInstanceUID.values, score=S, noul=N)
print("\nbar to clear: leehans 0.8932. Gold deltas under ~0.015 are noise.")
print("DONE", flush=True)
