"""Rules-based report -> 12-label extractor (approach: "rules").

Baseline for the label-derivation problem: 58 of 4407 training studies carry
hand labels, the rest must be derived from their radiology report. This is the
number every later approach has to beat.

Method: split each report into clauses, look for a finding term for each label,
require a qualifier where the bare noun is not enough (a meniscus is only
positive if a *tear* word is present), gate the medial/lateral labels on a
laterality word, then check the clause for negation. Reports are multilingual
(Latin/Greek/Cyrillic scripts, ~12 languages), so every term bank is a stem
list covering the languages actually observed in train.csv.

Emits graded scores rather than 0/1 so the metric (AUC) has something to rank.
Stdlib only -- must run inside an offline Kaggle notebook.
"""
import re
import unicodedata

from rsna_labels import LABELS, load_train, write_derived

# ---------------------------------------------------------------- normalizing

def norm(text):
    """Lowercase and strip combining accents. Handles Spanish/Portuguese
    diacritics and Greek tonos in one pass; Cyrillic is unaffected.

    NFKD, not NFD, and that is not a detail: 305 of the 321 Greek-script
    reports type mu as MICRO SIGN U+00B5 rather than GREEK SMALL LETTER MU
    U+03BC. NFD leaves the two distinct, so every Greek term containing a mu
    (menisc-, oedema, katagma) silently fails to match. NFKD folds them.

    Turkish needs one more fold. Its ş/ğ/ü/ö/ç all decompose to an ASCII base
    plus a combining mark and so come out as s/g/u/o/c above, but DOTLESS I
    U+0131 is a letter in its own right with no decomposition, so it survives
    untouched. Left alone it splits every Turkish stem in two spellings
    (yirtik/yırtık, kikirdak/kıkırdak) and the term bank silently matches
    neither. Folding it to plain i is safe: no other language in this corpus
    uses the character, and no existing bank term contains it.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("ı", "i")
    return re.sub(r"\s+", " ", text)


# Sentence-ish split. Also breaks on bullets, newlines and contrastive
# conjunctions, so "no meniscal tear but the ACL is torn" becomes two clauses
# and the negation does not leak across.
# NB: the contrastive conjunctions must be written in their own script.
# Spelling Bulgarian "но" as Latin "no" made this regex split every Spanish
# "No hay ..." in two, stranding the negation in a discarded fragment and
# turning "No hay quistes popliteos" into a positive Baker's.
CLAUSE_SPLIT = re.compile(
    r"[.;:\n\r>*•]|(?<=\s)(?:but|however|although|though|whereas|"
    r"pero|sin embargo|aunque|"
    r"maar|echter|"
    r"ομως|αλλα|"          # greek
    r"но|ала|обаче"        # bulgarian
    r")(?=\s)"
)


def clauses(text):
    return [c.strip() for c in CLAUSE_SPLIT.split(norm(text)) if c and c.strip()]


# ---------------------------------------------------------------- term banks
# Stems, matched as substrings after normalization. Grouped by concept so the
# same bank can serve several labels.

NEGATION = [
    # english
    "no ", "not ", "without", "absent", "negative for", "unremarkable",
    "intact", "preserved", "normal", "no evidence", "rules out", "free of",
    # spanish / portuguese
    "sin ", "sem ", "no hay", "no se", "ausencia", "conservad", "integr",
    "normal", "descarta",
    # dutch / german
    "geen", "zonder", "onopvallend", "kein", "ohne", "unauffallig",
    # italian / french
    "assenza", "nessun", "non si", "pas de", "absence", "sans ",
    # greek
    "χωρις", "δεν ", "φυσιολογικ", "ουδεν", "ακεραι", "εντος του φυσιολογικου",
    # bulgarian / russian
    "няма", "без ", "не се", "нормал", "запазен", "б.о", "липсва", "интактен",
    # turkish -- NB these are verb-final, so they land AFTER the finding and
    # are carried by the post-modifier list in negated(); they are kept here
    # too for the clauses where the cue does lead.
    "yok", "izlenmemis", "saptanmamis", "gorulmemis", "mevcut degil",
    "normaldir", "dogal", "korunmus", "intakt", "belirgin degil",
]

TEAR = [
    "tear", "torn", "rupture", "ruptur", "disrupt", "fissur",
    "rotura", "ruptura", "rotur", "desgarro", "lesion",
    "scheur", "riss", "rottura", "dechirure",
    "ρηξη", "ρωγμ", "σχισ",
    "руптур", "скъсван", "разкъсв", "увреда", "разрив",
    # turkish: yirtik/yirtigi (tear), butunluk kaybi (loss of continuity),
    # kopma (avulsion). "ruptur" above already covers ruptur/ruptu:re.
    "yirtik", "yirtig", "butunluk kayb", "kopma", "yirtilma",
]

MENISCUS = ["menisc", "menisk", "μηνισκ", "менисk", "мениск"]

# Turkish spells lateral the Latin way but medial as "medyal", and uses ic/dis
# (inner/outer) for the collaterals. Missing "medyal" is why Medial Meniscus
# read as 96% never-mentioned in Turkish while Lateral Meniscus read as 49%:
# the compartment gate fired on one side of the same sentence and not the other.
MEDIAL = ["medial", "interno", "interna", "mediaal", "mediale", "εσω", "медиал", "вътреш",
          "medyal", "ic yan"]
LATERAL = ["lateral", "externo", "externa", "lateraal", "laterale", "εξω", "латерал", "външ",
           "dis yan"]

CARTILAGE_LOSS = [
    "cartilage loss", "chondral", "chondrosis", "chondropath", "chondromalac",
    "cartilage defect", "cartilage thinning", "cartilage heterogeneity",
    "osteophyt", "osteoarthrit", "arthros", "arthrit", "degenerative change",
    "joint space narrowing", "subchondral",
    "condropat", "condral", "artrosis", "gonartrosis", "artrose", "artrosi",
    "osteofit", "cartilago adelgaz", "perdida de cartilago",
    "kraakbeen", "knorpel",
    "χονδροπαθ", "χονδρ", "αρθριτιδ", "αρθροπαθ", "οστεοφυτ", "εκφυλιστικ",
    "хрущял", "гонартроз", "остеоартрит", "остеофит", "изтънен", "артроз",
    # turkish. "artroz"/"gonartroz" end in -z where the romance forms end in
    # -s, so the existing "artros" stem does not reach them.
    "kondropati", "kondral", "kikirdak", "artroz", "gonartroz", "osteoartrit",
    "dejeneratif", "eklem aralig", "kondromalazi", "subkondral",
]

PATELLOFEMORAL = [
    "patellofemoral", "patellar", "patella", "trochlea", "trochlear",
    "rotulian", "rotula", "femoropatelar", "patelofemoral", "troclea",
    "επιγονατιδ", "τροχιλ",
    "пател", "епигонатид",
    "pateller", "patellofemoral", "troklea", "diz kapag",
]

EFFUSION = [
    "effusion", "joint fluid", "fluid in the joint", "hydrops", "hydrarthros",
    "suprapatellar fluid", "fluid accumulat",
    "derrame", "liquido articular",
    "vocht", "erguss",
    "versamento", "epanchement",
    "υδραρθρ", "συλλογη υγρου", "ενδαρθρικη συλλογη",
    "ставен излив", "излив", "изливи",
    # turkish
    "efuzyon", "sivi artis", "eklem sivisi", "sivi miktarinda artis",
    "eklem ici sivi", "hidrops",
]

SYNOVITIS = [
    "synovitis", "synovial thicken", "synovial proliferation", "synovial hypertroph",
    "sinovitis", "sinovite", "synovite", "synovial membrane thicken",
    "υμενιτιδ", "αρθρικου υμενα",
    "синовит", "синовиал",
    # turkish "sinovit" (23 reports, every one a genuine synovitis statement --
    # "diz ekleminde sinovit"). Deliberately NOT "sinovyal": that is dominated
    # by "sinovyal plika" and "sinovyal kist", which are a normal anatomic
    # variant and a cyst, not synovitis.
    "sinovit", "sinovyal hipertrofi", "sinovyal kalinlas",
    # german, seen as "reizsynovialitis" / "synoviale proliferationen" /
    # "verdickter synovia"
    "synovialitis", "synoviale proliferation", "synovialen proliferation",
    "verdickter synovia",
]

# Baker's is named directly, or is "cyst" + "popliteal" in the same clause.
# A bare "poplite*" stem is useless here: across train.csv it appears 2052
# times as popliteal vein/artery/popliteus tendon against only 184 in a cyst
# context, so matching it alone triples the derived prevalence.
BAKER_NAME = ["baker", "бекеров", "бейкър"]
CYST = ["cyst", "quist", "cisto", "cisti", "kyste", "zyste", "κυστ", "киста",
        "kist"]
POPLITEAL = ["poplite", "popliteo", "ignyak", "ιγνυακ", "поплите"]

FRACTURE = [
    "fracture", "fractura", "fratura", "frattura", "fraktur", "fractuur",
    "καταγμα", "фрактур", "счупван",
    "kirik",          # turkish; "fraktur" above already covers the loanword
]

CONTUSION = [
    "contusion", "bone bruise", "bone marrow edema", "bone marrow oedema",
    "marrow edema", "marrow oedema", "bone edema", "trabecular edema",
    "contusion osea", "edema oseo", "edema de medula osea",
    "botoedeem", "knochenmarkodem", "edema osseo",
    "οιδημα μυελου", "οστικο οιδημα", "μυελου των οστων",
    "костномозъчен едем", "костен едем", "контузи",
    # turkish: "kemik iligi odemi" is the standard phrasing
    "kemik iligi odem", "kemik odem", "kontuzyon", "iligi odem",
    "trabekuler odem", "ezilme",
]

ACL = [
    "acl", "anterior cruciate", "cruzado anterior", "ligamento cruzado anterior",
    "voorste kruisband", "vorderes kreuzband", "crociato anteriore",
    "croise anterieur", "lca",
    "προσθιος χιαστος", "προσθιου χιαστου",
    "предна кръстна", "предната кръстна",
    # turkish. Anterior only, on purpose: a bare "capraz bag" also names the
    # posterior cruciate ("arka capraz bag"), and these reports discuss both
    # in the same sentence, so the generic stem would attribute PCL findings
    # to the ACL.
    "on capraz", "anterior capraz", "ocb",
]

MCL = [
    "mcl", "medial collateral", "colateral medial", "colateral interno",
    "ligamento colateral medial", "mediale collaterale", "innenband",
    "collaterale mediale", "collateral medial",
    "εσω πλαγιο", "εσω πλαγιος",
    "медиален колатерал", "вътрешен колатерал", "медиалния колатерал",
    "medial kollateral", "medyal kollateral", "ic yan bag",
]

COLLATERAL_GENERIC = ["collateral", "colateral", "πλαγι", "колатерал",
                      "kollateral", "yan bag", "collaterale", "kollateralband"]

CRUCIATE_GENERIC = [
    "cruciate", "cruzados", "ligamentos cruzados", "χιαστ", "kruisband",
    "kreuzband", "crociat", "croise", "кръстн", "capraz bag", "capraz baglar",
]


def has(clause, terms):
    return any(t in clause for t in terms)


def negated(clause, terms):
    """True if a negation cue governs the finding term in this clause.

    Radiology negation is overwhelmingly pre-modifying ("no meniscal tear",
    "sin rotura"), so a cue counts when it appears before the finding term.
    Post-modifiers like 'intact'/'preserved'/'normal' are also accepted
    anywhere in the clause, since 'the ACL is intact' puts them after.

    Turkish depends on that second path entirely. It is verb-final, so the
    negation is the last word of the clause every time -- "yirtik
    izlenmemistir" is literally "tear not-observed", and "capraz baglar
    normaldir" is "cruciate ligaments are-normal". Scanning only the text
    before the finding term would score both as positive findings, which is
    worse than not covering Turkish at all.
    """
    idx = min((clause.find(t) for t in terms if t in clause), default=-1)
    if idx < 0:
        return False
    before = clause[:idx]
    if any(c in before for c in NEGATION):
        return True
    post = ("intact", "preserved", "normal", "unremarkable", "conservad",
            "integr", "φυσιολογικ", "ακεραι", "запазен", "нормал", "б.о",
            "intact", "onopvallend",
            # turkish, verb-final
            "izlenmemis", "saptanmamis", "gorulmemis", "mevcut degil",
            "normaldir", "dogal", "korunmus", "intakt", "yoktur", "yok",
            "belirgin degil", "izlenmedi", "salim")
    return any(p in clause[idx:] for p in post)


# ------------------------------------------------------------------ per-label

INTACT = [
    "intact", "preserved", "normal", "unremarkable", "conservad", "integr",
    "sin alteraciones", "no torn", "not torn", "onopvallend", "unauffallig",
    "φυσιολογικ", "ακεραι", "εντος του φυσιολογικου", "χωρις παθολογικ",
    "запазен", "нормал", "б.о", "интактен", "непроменен",
]


def _structure_finding(cls, structure_terms, evidence_terms, side_terms=None,
                       generic_terms=None):
    """Evidence for a named anatomical structure.

    Split from the finding term on purpose: 1581 reports name the ACL and 680
    of those say it is intact/normal/preserved *without using any tear word*.
    Requiring a tear term to fire at all scored those as 'never mentioned'
    (0.25) instead of the confident negative they are, which is why ligament
    coverage read as ~90% missing in v1.

    `generic_terms` handles the unsided plural: "ligamentos cruzados y
    colaterales dentro de limites normales", "οι πλαγιοι και οι χιαστοι
    συνδεσμοι εντος του φυσιολογικου". 541 reports clear both collaterals or
    both cruciates in one breath without ever naming a side, and every one of
    them was being scored "never mentioned" rather than the confident negative
    it is.

    Two deliberate restrictions, because an unsided mention is ambiguous:

      * negatives only. "Both collaterals normal" pins the MCL; "collateral
        ligament tear" does not say *which* collateral, so it may never add
        positive evidence.
      * only when the specific bank found nothing at all. Otherwise a report
        that names a torn medial collateral and elsewhere calls the
        collaterals unremarkable would drop from 0.90 to a hedged 0.65 on the
        strength of the vaguer sentence.
    """
    pos = neg = 0
    for c in cls:
        if side_terms and not has(c, side_terms):
            continue
        if not has(c, structure_terms):
            continue
        if has(c, evidence_terms):
            if negated(c, evidence_terms):
                neg += 1
            else:
                pos += 1
        elif has(c, INTACT) or any(n in c for n in NEGATION):
            neg += 1

    if generic_terms and not pos and not neg:
        for c in cls:
            if not has(c, generic_terms) or has(c, evidence_terms):
                continue
            if has(c, INTACT) or any(n in c for n in NEGATION):
                neg += 1
    return pos, neg


def _side_finding(cls, side_terms, finding_terms, require=None):
    """Positive evidence for a compartment-specific finding: the clause must
    name the side, carry the finding, optionally carry a qualifier, and not be
    negated."""
    pos = neg = 0
    for c in cls:
        if not has(c, side_terms) or not has(c, finding_terms):
            continue
        if require and not has(c, require):
            continue
        if negated(c, finding_terms) or (require and negated(c, require)):
            neg += 1
        else:
            pos += 1
    return pos, neg


def _bakers(cls):
    pos = neg = 0
    for c in cls:
        named = has(c, BAKER_NAME) or (has(c, CYST) and has(c, POPLITEAL))
        if not named:
            continue
        key = BAKER_NAME if has(c, BAKER_NAME) else CYST
        if negated(c, key):
            neg += 1
        else:
            pos += 1
    return pos, neg


def _simple_finding(cls, finding_terms):
    pos = neg = 0
    for c in cls:
        if not has(c, finding_terms):
            continue
        if negated(c, finding_terms):
            neg += 1
        else:
            pos += 1
    return pos, neg


def score(pos, neg):
    """Graded confidence so AUC has something to rank.

    Never emit hard 0/1: the gold labels themselves are inconsistent with
    report wording (see the README), so hedged scores are honest.
    """
    if pos and not neg:
        return 0.90
    if pos and neg:
        return 0.65          # mentioned both ways -- weak positive
    if neg and not pos:
        return 0.08          # explicitly denied
    return 0.25              # never mentioned; absence is the base rate


def extract(report):
    cls = clauses(report)
    out = {}

    out["Medial Meniscus"] = score(*_structure_finding(cls, MENISCUS, TEAR, MEDIAL))
    out["Lateral Meniscus"] = score(*_structure_finding(cls, MENISCUS, TEAR, LATERAL))
    out["Medial OA"] = score(*_side_finding(cls, MEDIAL, CARTILAGE_LOSS))
    out["Lateral OA"] = score(*_side_finding(cls, LATERAL, CARTILAGE_LOSS))
    out["PF OA"] = score(*_side_finding(cls, PATELLOFEMORAL, CARTILAGE_LOSS))

    out["ACL"] = score(*_structure_finding(cls, ACL, TEAR,
                                           generic_terms=CRUCIATE_GENERIC))
    out["MCL"] = score(*_structure_finding(cls, MCL, TEAR,
                                           generic_terms=COLLATERAL_GENERIC))

    out["Effusion"] = score(*_simple_finding(cls, EFFUSION))
    out["Synovitis"] = score(*_simple_finding(cls, SYNOVITIS))
    out["Baker's"] = score(*_bakers(cls))
    out["Fracture"] = score(*_simple_finding(cls, FRACTURE))
    out["Contusion"] = score(*_simple_finding(cls, CONTUSION))

    return out


def main():
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "rules"
    rows = load_train()
    records = [(r["StudyInstanceUID"], extract(r["Report"])) for r in rows]
    path = write_derived(name, records)
    print(f"wrote {len(records)} studies -> {path}")


if __name__ == "__main__":
    main()
