"""Write-up figures as dependency-free SVG.

Usage:  python tools/make_figures.py [output_dir]   (default: writeup/assets)

No competition images are used anywhere: the rules forbid redistributing the
data, so every figure is drawn from our own measured numbers.
Palette: dataviz reference categorical slots 1-2 (validated, light mode).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "writeup", "assets")
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FONT = "font-family='system-ui,-apple-system,Segoe UI,Roboto,sans-serif'"


def svg(w, h, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' "
            f"viewBox='0 0 {w} {h}' {FONT}>"
            f"<rect width='{w}' height='{h}' rx='12' fill='{SURFACE}'/>{body}</svg>")


def text(x, y, s, size: float = 13, fill=INK2, anchor="start", weight=400, extra=""):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return (f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' "
            f"text-anchor='{anchor}' font-weight='{weight}' {extra}>{s}</text>")


# ---------------------------------------------------------------- progression
STEPS = [
    ("64px thumbnails", .605, .586), ("geometry normalisation", .669, .651),
    ("DINOv2 features", .713, .691), ("graded targets", .742, .710),
    ("fine-tuned + attention head", .795, .765), ("Turkish label repair", .807, .781),
    ("rules extractor v5", .827, .804), ("public LLM labels", .870, .853),
    ("3-model ensemble", .880, .868), ("+ RadImageNet ResNet-50", .896, .880),
    ("4-model, layer3 unfrozen", .907, .888), ("+10% training data", .908, .888),
    ("5-model hedge (final)", .910, .891),
]


def progression():
    W, H = 960, 470
    L, R, T, B = 120, 150, 64, 150
    pw, ph = W - L - R, H - T - B
    ymin, ymax = .55, .98
    X = lambda i: L + pw * i / (len(STEPS) - 1)
    Y = lambda v: T + ph * (ymax - v) / (ymax - ymin)
    out = [text(L, 32, "Score by submission, public vs private leaderboard", 17, INK, weight=600),
           text(L, 52, "Macro ROC-AUC. Each point changes one thing; the private board is 70% of the test set.", 12.5, INK2)]
    for v in (.6, .7, .8, .9):
        out.append(f"<line x1='{L}' x2='{L+pw}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='{GRID}' stroke-width='1'/>")
        out.append(text(L - 10, Y(v) + 4, f"{v:.1f}", 12, MUTED, "end"))
    # winner reference
    yw = Y(.960)
    out.append(f"<line x1='{L}' x2='{L+pw}' y1='{yw:.1f}' y2='{yw:.1f}' stroke='{MUTED}' stroke-width='1.5' stroke-dasharray='5 5'/>")
    out.append(text(L + pw + 10, yw + 4, "1st place, private 0.960", 12, INK2))
    for key, col, lab in ((1, BLUE, "public"), (2, ORANGE, "private")):
        pts = " ".join(f"{X(i):.1f},{Y(s[key]):.1f}" for i, s in enumerate(STEPS))
        out.append(f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2' stroke-linejoin='round'/>")
        for i, s in enumerate(STEPS):
            out.append(f"<circle cx='{X(i):.1f}' cy='{Y(s[key]):.1f}' r='4.5' fill='{col}' stroke='{SURFACE}' stroke-width='2'/>")
        last = STEPS[-1][key]
        dy = -6 if key == 1 else 14
        out.append(text(X(len(STEPS) - 1) + 12, Y(last) + dy, f"{lab} {last:.3f}", 13, INK, weight=600))
    for i, s in enumerate(STEPS):
        x = X(i)
        out.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{T+ph}' y2='{T+ph+5}' stroke='{MUTED}'/>")
        out.append(text(x, T + ph + 16, s[0], 11.5, INK2, "end",
                        extra=f"transform='rotate(-38 {x:.1f} {T+ph+16})'"))
    # legend (always present for 2 series)
    lx = L + pw - 180
    for j, (col, lab) in enumerate(((BLUE, "public LB (30%)"), (ORANGE, "private LB (70%)"))):
        out.append(f"<line x1='{lx}' x2='{lx+18}' y1='{T+ph-58+j*20}' y2='{T+ph-58+j*20}' stroke='{col}' stroke-width='3'/>")
        out.append(text(lx + 26, T + ph - 54 + j * 20, lab, 12.5, INK2))
    return svg(W, H, "".join(out))


# ------------------------------------------------------------------- teachers
TEACHERS = [  # gold-58 macro AUC (57 common studies for the public sources)
    ("1st place: LLM agents reading the IMAGES", .936, ORANGE),
    ("flight0234 hybrid (best public)", .8999, BLUE),
    ("stevenleehans v4_blend", .8932, BLUE),
    ("stevenleehans v2", .8874, BLUE),
    ("stevenleehans full", .8777, BLUE),
    ("riadmohamed42 jev-knee", .8745, BLUE),
    ("pilkwang LLM labels", .8700, BLUE),
    ("lixin73 GPT-5.6-Sol", .8347, BLUE),
    ("laymond Qwen weak labels", .8114, BLUE),
    ("our multilingual rules extractor", .7896, BLUE),
]


def teachers():
    W, H = 960, 470
    L, R, T = 300, 70, 82
    bh, gap = 24, 9
    xmin, xmax = .75, .96
    pw = W - L - R
    X = lambda v: L + pw * (v - xmin) / (xmax - xmin)
    out = [text(32, 32, "How good can labels be? Every label source scored on the 58 hand-labelled studies", 17, INK, weight=600),
           text(32, 54, "Macro ROC-AUC. Nine sources that read the REPORT all stop near 0.90; reading the images is what broke it.", 12.5, INK2)]
    bottom = T + len(TEACHERS) * (bh + gap)
    for v in (.75, .80, .85, .90, .95):
        out.append(f"<line x1='{X(v):.1f}' x2='{X(v):.1f}' y1='{T-8}' y2='{bottom}' stroke='{GRID}'/>")
        out.append(text(X(v), bottom + 18, f"{v:.2f}", 12, MUTED, "middle"))
    x9 = X(.90)
    out.append(f"<line x1='{x9:.1f}' x2='{x9:.1f}' y1='{T-14}' y2='{bottom}' stroke='{MUTED}' stroke-width='1.5' stroke-dasharray='5 5'/>")
    out.append(text(x9 + 6, T - 18, "report ceiling ~0.90", 12, INK2))
    for i, (name, v, col) in enumerate(TEACHERS):
        y = T + i * (bh + gap)
        cy = y + bh / 2
        # dot plot, not bars: the axis starts at 0.75, and a bar's length would
        # imply a zero baseline and exaggerate the differences
        out.append(f"<line x1='{L}' x2='{L+pw}' y1='{cy:.1f}' y2='{cy:.1f}' stroke='{GRID}' stroke-width='1'/>")
        out.append(f"<circle cx='{X(v):.1f}' cy='{cy:.1f}' r='7' fill='{col}' stroke='{SURFACE}' stroke-width='2'/>")
        out.append(text(L - 10, y + bh / 2 + 4.5, name, 12.5, INK if col == ORANGE else INK2, "end",
                        weight=600 if col == ORANGE else 400))
        out.append(text(X(v) + 14, y + bh / 2 + 4.5, f"{v:.3f}", 12.5, INK, weight=600))
    ly = bottom + 46
    for j, (col, lab) in enumerate(((BLUE, "reads the radiology report (public datasets / ours)"),
                                    (ORANGE, "reads the MRI images + report (winning team)"))):
        x0 = 32 + j * 430
        out.append(f"<circle cx='{x0+7}' cy='{ly-3}' r='7' fill='{col}'/>")
        out.append(text(x0 + 22, ly + 2, lab, 12.5, INK2))
    return svg(W, H + 20, "".join(out))


# ------------------------------------------------------------------- pipeline
def pipeline():
    W, H = 960, 340
    out = [text(32, 34, "Our pipeline (public 0.910 / private 0.891)", 17, INK, weight=600)]

    def box(x, y, w, h, title, sub, col=INK2):
        out.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='10' fill='#ffffff' stroke='{GRID}' stroke-width='1.5'/>")
        out.append(f"<rect x='{x}' y='{y}' width='4' height='{h}' rx='2' fill='{col}'/>")
        out.append(text(x + 16, y + 24, title, 13.5, INK, weight=600))
        for k, line in enumerate(sub):
            out.append(text(x + 16, y + 44 + k * 17, line, 12, INK2))

    def arrow(x1, y1, x2, y2):
        if y1 == y2:   # horizontal, pointing right
            out.append(f"<line x1='{x1}' y1='{y1}' x2='{x2-7}' y2='{y2}' stroke='{MUTED}' stroke-width='1.5'/>")
            out.append(f"<path d='M{x2},{y2} l-8,-4.5 v9 z' fill='{MUTED}'/>")
        else:          # vertical, pointing down
            out.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2-7}' stroke='{MUTED}' stroke-width='1.5'/>")
            out.append(f"<path d='M{x2},{y2} l-4.5,-8 h9 z' fill='{MUTED}'/>")

    y1, h = 62, 110
    box(32, y1, 200, h, "MRI study (DICOM)", ["3 planes, 1 series each", "sagittal / coronal / axial"])
    box(262, y1, 210, h, "Geometry normalisation", ["sort slices by position", "crop 160 mm of anatomy", "flip every knee one way"], BLUE)
    box(502, y1, 200, h, "Encoder per slice", ["12 slices x 3 planes, 224px", "DINOv2 or RadImageNet", "ResNet-50, fine-tuned"], BLUE)
    box(732, y1, 196, h, "Per-finding attention", ["each of 12 findings picks", "its own slices", "-> 12 probabilities"], BLUE)
    for a, b in ((232, 262), (472, 502), (702, 732)):
        arrow(a, y1 + h / 2, b, y1 + h / 2)

    y2 = 210
    box(32, y2, 200, h, "Radiology reports", ["train set only, ~12 languages", "test studies have none"])
    box(262, y2, 210, h, "Labels from text", ["public LLM-read labels", "graded soft targets", "(only 58 hand labels exist)"], ORANGE)
    box(502, y2, 200, h, "Train, select on gold-58", ["BCE on soft labels", "keep best epoch by the", "58 hand-labelled studies"], ORANGE)
    box(732, y2, 196, h, "Ensemble of 5", ["probability average", "members picked by low", "correlation, on GPU"], BLUE)
    arrow(232, y2 + h / 2, 262, y2 + h / 2)
    arrow(472, y2 + h / 2, 502, y2 + h / 2)
    # training feeds the models; models feed the ensemble
    out.append(f"<path d='M602,{y2} V{y1+h+8}' stroke='{MUTED}' stroke-width='1.5' fill='none' stroke-dasharray='4 4'/>")
    out.append(text(612, (y2 + y1 + h) / 2 + 4, "trains", 11.5, MUTED))
    arrow(830, y1 + h, 830, y2)
    return svg(W, H, "".join(out))


for name, fn in (("progression", progression), ("label_ceiling", teachers), ("pipeline", pipeline)):
    p = os.path.join(OUT, f"{name}.svg")
    open(p, "w").write(fn())
    print("wrote", p)
