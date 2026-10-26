"""Extractor coverage per *language*, not per script.

`evaluate.py` buckets reports as Latin / Greek / Cyrillic. That hides the thing
that matters: Turkish, Spanish, German, Dutch, Portuguese and English are all
"Latin", and they share almost no medical vocabulary. Turkish writes menisküs /
çapraz bağ / yırtık / kıkırdak; Spanish writes menisco / ligamento cruzado /
rotura / cartílago. A term bank rich in one and empty in the other averages out
to a middling Latin number that looks like noise rather than a hole.

Language is identified by stopwords and orthography, which is crude but does not
need a dependency and is unambiguous at report length. Then, per language, the
fraction of reports where the extractor found *nothing at all*: a report where
all twelve labels come back "not mentioned" is almost always a coverage failure,
since a knee MRI report says something about the knee.

Usage:  .venv/bin/python src/probe_language.py
"""
import sys

import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import extract_rules as R
from rsna_labels import LABELS

# distinctive, high-frequency markers; first match wins, so order matters
# (Turkish before the general Latin languages, since it is the one at risk)
MARKERS = [
    ("Greek",      ["ευρηματα", "γονατοσ", "μηνισκ", "εξετασ", "αρθρ"]),
    ("Bulgarian",  ["мр", "коляно", "става", "менискус", "находка", "връзка"]),
    ("Thai",       ["ข้อ", "เข่า"]),
    ("Turkish",    ["menisk", "diz mrg", "capraz bag", "yirtik", "kikirdak",
                    "izlenmistir", "mevcuttur", "eklem", "bulgular",
                    "tetkik", "dejenerasyon", "sinyal"]),
    ("Spanish",    ["rodilla", "menisco", "tecnica: rmn", "resultados",
                    "rotura", "derrame", "ligamento cruzado"]),
    ("Portuguese", ["joelho", "menisco medial", "lesao", "achados"]),
    ("German",     ["kniegelenk", "meniskus", "gelenkerguss", "bandapparat",
                    "knorpel", "hinterhorn", "befund"]),
    ("Dutch",      ["knie", "meniscus mediaal", "bevindingen", "kraakbeen",
                    "vraagstelling"]),
    ("Italian",    ["ginocchio", "menisco mediale", "versamento", "reperti"]),
    ("French",     ["genou", "menisque", "epanchement", "ligament croise"]),
    ("English",    ["knee", "meniscus", "tear", "findings", "technique",
                    "effusion", "intact"]),
]


def language(t):
    for name, keys in MARKERS:
        if any(k in t for k in keys):
            return name
    return "unknown"


df = pd.read_csv("data/train.csv")
texts = [R.norm(str(t)) for t in df.Report]
langs = [language(t) for t in texts]
df = df.assign(lang=langs)

VERSION = sys.argv[1] if len(sys.argv) > 1 else "rules_v3"
derived = pd.read_csv(f"derived/labels_{VERSION}.csv").set_index("StudyInstanceUID")
print(f"labels: {VERSION}\n")
derived = derived.loc[df.StudyInstanceUID]

# 0.25 is the extractor's "not mentioned" score; anything else is an opinion
UNMENTIONED = 0.25
opinions = (derived[LABELS].values != UNMENTIONED).sum(axis=1)
df = df.assign(n_findings=opinions)

print("=" * 74)
print("EXTRACTOR COVERAGE BY LANGUAGE")
print("=" * 74)
print(f"  {'language':<13}{'reports':>9}{'silent':>9}{'silent%':>9}"
      f"{'mean findings/report':>22}")
rows = []
for lang, g in df.groupby("lang"):
    silent = int((g.n_findings == 0).sum())
    rows.append((len(g), lang, silent, silent / len(g), g.n_findings.mean()))
for n, lang, silent, frac, mean in sorted(rows, reverse=True):
    flag = "   <-- HOLE" if frac > 0.25 and n > 40 else ""
    print(f"  {lang:<13}{n:>9}{silent:>9}{frac:>8.0%}{mean:>22.2f}{flag}")

print(f"\n  {'TOTAL':<13}{len(df):>9}{int((df.n_findings==0).sum()):>9}"
      f"{(df.n_findings==0).mean():>8.0%}{df.n_findings.mean():>22.2f}")

print("\n" + "=" * 74)
print("PER-LABEL 'never mentioned' RATE, BY LANGUAGE  (high = a term-bank hole)")
print("=" * 74)
big = [lang for n, lang, _, _, _ in sorted(rows, reverse=True) if n >= 40]
print(f"  {'label':<19}" + "".join(f"{l[:7]:>9}" for l in big))
for lab in LABELS:
    line = f"  {lab:<19}"
    for lang in big:
        sel = df.lang == lang
        rate = (derived[lab].values[sel.values] == UNMENTIONED).mean()
        line += f"{rate:>8.0%} "
    print(line)

print("\nA knee MRI report says something about the knee. A language whose")
print("reports come back silent is a vocabulary gap, not a healthy cohort.")
