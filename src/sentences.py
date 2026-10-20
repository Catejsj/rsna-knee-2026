"""Split the 4407 reports into individual finding sentences.

This is the correction to "cluster the reports": a report is not about one
thing. The labeled studies carry a median of ~3 positive findings out of 12,
so two reports can share "torn medial meniscus" while differing everywhere
else. Clustering whole reports groups them by overall profile, which is not
what we want to attach a label to.

A *sentence* usually states exactly one finding:

    "Rotura de menisco interno."
    "Скъсване на медиалния менискус."
    "Medial meniscus tear."

Those three should land in one cluster across three languages. That is the
unit the whole plan depends on.

Output keeps the ORIGINAL text (accents, script intact) -- the embedding model
needs real text, unlike the rules extractor which works on a folded form.
"""
import csv
import os
import re

from rsna_labels import DATA, ROOT, load_train

# Split on sentence enders and on the bullet/newline styles used across sites.
SPLIT = re.compile(r"(?<=[.;!?])\s+|[\n\r]+|(?:^|\s)[>*•\-]\s+")

# Lines that are headers or acquisition detail, not findings. Matched against
# the lowercased sentence; these carry no diagnostic content and would only
# add dense, meaningless clusters.
NOISE = re.compile(
    r"^(technique|tecnica|técnica|findings|hallazgos|impression|impresion|"
    r"impresión|exam type|exam date|indication|comparison|history|"
    r"clinical|antecedentes|protocol|scanprotocol|conclusion|conclusión|"
    r"bevindingen|ευρημα|ευρήμα|τεχνικη|τεχνική|заключение|мр находка)"
    r"[\s:]*$",
    re.I,
)

# Sequence/protocol chatter: "sag pd fs", "3 planes", "1.5 tesla", "[DATE]".
PROTOCOL = re.compile(
    r"\b(t1|t2|pd|stir|fs|fse|tse|sag|cor|ax|axial|sagittal|coronal|tesla|"
    r"sequence|secuencia|ακολουθ|mri|rmn|mr\b)\b", re.I
)


def is_finding(sentence):
    """Keep sentences that plausibly state a finding."""
    s = sentence.strip()
    if len(s) < 12 or len(s) > 400:
        return False
    if NOISE.match(s):
        return False
    letters = sum(c.isalpha() for c in s)
    if letters < len(s) * 0.5:          # mostly numbers/punctuation
        return False
    # Protocol lines are dense in sequence tokens and short on prose.
    hits = len(PROTOCOL.findall(s))
    if hits >= 3 and len(s.split()) < 25:
        return False
    return True


def script_of(text):
    for ch in text:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF:
            return "Cyrillic"
        if 0x0370 <= o <= 0x03FF:
            return "Greek"
    return "Latin"


def split_report(report):
    out = []
    for part in SPLIT.split(report):
        if part is None:
            continue
        part = re.sub(r"\s+", " ", part).strip(" .;:-*>•")
        if part and is_finding(part):
            out.append(part)
    return out


def main():
    rows = load_train()
    path = os.path.join(DATA, "sentences.csv")
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["StudyInstanceUID", "sent_idx", "script", "text"])
        for r in rows:
            for i, s in enumerate(split_report(r["Report"])):
                w.writerow([r["StudyInstanceUID"], i, script_of(s), s])
                n += 1
    print(f"{n} finding sentences from {len(rows)} reports -> {path}")
    print(f"mean {n/len(rows):.1f} sentences per report")


if __name__ == "__main__":
    main()
