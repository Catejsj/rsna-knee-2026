"""Shared constants and IO for the RSNA knee label pipeline.

Stdlib only, on purpose: this has to run unchanged inside a Kaggle notebook
with internet disabled, so it must not depend on anything being installed.
"""
import csv
import os
import sys

csv.field_size_limit(sys.maxsize)

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DERIVED = os.path.join(ROOT, "derived")


def load_train():
    """All 4407 training studies: dicts with StudyInstanceUID, Report, and the
    12 label columns (empty string where unlabeled)."""
    with open(os.path.join(DATA, "train.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def gold_rows(rows):
    """The hand-labeled subset (58 studies). Values are '0'/'1'."""
    return [r for r in rows if r["ACL"].strip() != ""]


def write_derived(name, records):
    """records: list of (study_uid, {label: float score in [0,1]})."""
    os.makedirs(DERIVED, exist_ok=True)
    path = os.path.join(DERIVED, f"labels_{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["StudyInstanceUID"] + LABELS)
        for uid, scores in records:
            w.writerow([uid] + [f"{scores[l]:.4f}" for l in LABELS])
    return path


def read_derived(name):
    path = os.path.join(DERIVED, f"labels_{name}.csv")
    with open(path, encoding="utf-8") as fh:
        return {r["StudyInstanceUID"]: {l: float(r[l]) for l in LABELS}
                for r in csv.DictReader(fh)}
