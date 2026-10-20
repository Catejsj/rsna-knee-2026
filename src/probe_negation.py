"""Go/no-go test for the clustering plan.

The plan is: group sentences that mean the same thing across languages, then
push a known label to everyone in the group. That only works if the grouping
separates a finding from its denial.

Sentence embeddings are famously weak at negation -- "the meniscus is torn"
and "the meniscus is not torn" share almost every word. If they land close
together, label propagation will hand a torn-meniscus label to healthy knees
and poison the training set silently.

This measures it two ways before we build anything on top:
  A. cross-language:  does the same finding match across scripts? (must be HIGH)
  B. negation:        does a finding separate from its denial? (must be LOW)

If B is not clearly below A, clustering alone is unsafe and negation has to be
carried separately.
"""
import csv
import itertools
import sys

from sentence_transformers import SentenceTransformer, util

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/LaBSE",   # purpose-built for translation matching
]

# Sanity pairs: if these are not ~0.9+, the model is broken, not the domain.
SANITY = [
    ("Hello, how are you?", "Hola, cómo estás?"),
    ("Hello, how are you?", "Здравей, как си?"),
    ("Hello, how are you?", "Γεια σου, τι κάνεις;"),
]

# Same finding, five languages seen in the corpus.
CROSS = {
    "medial meniscus tear": [
        "Medial meniscus tear.",
        "Rotura del menisco interno.",
        "Скъсване на медиалния менискус.",
        "Ρήξη του έσω μηνίσκου.",
        "Scheur van de mediale meniscus.",
    ],
    "joint effusion": [
        "Joint effusion.",
        "Derrame articular.",
        "Ставен излив.",
        "Ενδαρθρική συλλογή υγρού.",
        "Gewrichtsvocht.",
    ],
}

# Finding vs its denial, same language.
NEG = [
    ("Medial meniscus tear.", "No medial meniscus tear."),
    ("The anterior cruciate ligament is torn.", "The anterior cruciate ligament is intact."),
    ("Joint effusion is present.", "There is no joint effusion."),
    ("Rotura del menisco interno.", "No hay rotura del menisco interno."),
    ("Ставен излив.", "Няма данни за ставен излив."),
    ("Ρήξη του έσω μηνίσκου.", "Χωρίς ρήξη του έσω μηνίσκου."),
    ("Baker's cyst is seen.", "No Baker's cyst."),
]


def run(model_name):
    print("=" * 70)
    print(f"MODEL: {model_name}")
    m = SentenceTransformer(model_name)
    print(f"device: {m.device}")

    print("\n0. SANITY  (everyday translations) -- must be ~0.9+, else model is broken")
    for a, b in SANITY:
        e = m.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        print(f"  {float(util.cos_sim(e[0], e[1])):.3f}   {a[:30]:<32}| {b[:30]}")

    print("\nA. CROSS-LANGUAGE  (same finding, different language) -- want HIGH")
    cross_scores = []
    for name, variants in CROSS.items():
        emb = m.encode(variants, convert_to_tensor=True, normalize_embeddings=True)
        sims = [float(util.cos_sim(emb[i], emb[j]))
                for i, j in itertools.combinations(range(len(variants)), 2)]
        cross_scores += sims
        print(f"  {name:<24} mean {sum(sims)/len(sims):.3f}   min {min(sims):.3f}")
    cross_mean = sum(cross_scores) / len(cross_scores)

    print("\nB. NEGATION  (finding vs its denial) -- want LOW")
    neg_scores = []
    for a, b in NEG:
        emb = m.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        s = float(util.cos_sim(emb[0], emb[1]))
        neg_scores.append(s)
        print(f"  {s:.3f}   {a[:42]:<44}| {b[:42]}")
    neg_mean = sum(neg_scores) / len(neg_scores)

    print(f"\n  cross-language {cross_mean:.3f} | negation {neg_mean:.3f} | "
          f"margin {cross_mean - neg_mean:+.3f}\n")
    return cross_mean, neg_mean


def main():
    results = {name: run(name) for name in MODELS}
    print("=" * 70)
    print("SUMMARY                       cross-lang   negation     margin")
    for name, (c, n) in results.items():
        print(f"  {name.split('/')[-1]:<28}{c:>9.3f}{n:>12.3f}{c - n:>+11.3f}")
    if all(n >= c for c, n in results.values()):
        print("\nVERDICT: on every model tested, a finding sits CLOSER to its own")
        print("denial than to the same finding in another language.")
        print("Clustering cannot separate asserted from denied on its own.")
        print("=> use clusters for WHICH concept, rules for ASSERTED vs DENIED.")


if __name__ == "__main__":
    sys.exit(main())
