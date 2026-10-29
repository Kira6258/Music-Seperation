"""Keeps the guide and the deck consistent with the corrected report (review fixes)."""
import sys
import docx
from pptx import Presentation
from pptx.chart.data import CategoryChartData

GUIDE, PPT = sys.argv[1], sys.argv[2]
GUIDE_OUT = sys.argv[3] if len(sys.argv) > 3 else GUIDE


def set_text(p, text):
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


# ───────────────────────── guide ─────────────────────────
d = docx.Document(GUIDE)
P = d.paragraphs


def find_par(startswith):
    for p in d.paragraphs:
        if p.text.startswith(startswith):
            return p
    raise KeyError(startswith)


set_text(find_par("Ideal-mask ceiling (oracle)"),
         "Ideal-mask oracle reference: the SI-SDR you get by applying the ideal ratio mask to the mixture and keeping the "
         "mixture phase. It is a reference point for mask models, not a strict upper bound.")
p = find_par("Purpose. A small CNN that answers")
set_text(p, "Purpose. A small CNN that answers \"which instruments are present?\" for a 6-second log-mel patch. Its network "
            "has eight outputs, but only six (vocals, drums, bass, violin, mridangam, ghatam) are trained. Guitar and piano "
            "have no ground truth, so their outputs are never trained and are not used.")
p = find_par("oracle = SI-SDR(mixture")
set_text(p, "oracle = SI-SDR(mixture × clamp(tgt/mix, 0, 1), target) using the true ratio mask: an oracle reference, not a "
            "strict upper bound for mask models.")
set_text(find_par("Violin separation is modest"),
         "Violin separation is modest (≈ 1 dB) on mixed recordings; its ideal-mask oracle reference is only 3.8 dB.")
set_text(find_par("Magnitude masks reuse the mixture phase"),
         "Magnitude masks reuse the mixture phase, which limits quality; the ideal-mask oracle reference is about 4–10 dB "
         "depending on the stem.")

for t in d.tables:
    heads = [c.text for c in t.rows[0].cells]
    if heads and heads[0] == "Dataset / stem":
        for r in list(t.rows):
            if r.cells[0].text.strip() in ("MUSDB piano", "MUSDB guitar"):
                r._tr.getparent().remove(r._tr)
        set_text(t.rows[0].cells[3].paragraphs[0], "Ideal-mask oracle")
    for r in t.rows:
        c0 = r.cells[0].text.strip()
        if c0 == "Absent-query training" and len(r.cells) >= 3:
            set_text(r.cells[2].paragraphs[0],
                     "Measured on four MUSDB test tracks with the final system: absent violin and mridangam come out 10 to "
                     "17 dB below the mix on three tracks and 38 dB below on one, and ghatam is silent (the model returns "
                     "silence for ghatam even when present). The earlier 27 to 178 dB figure was not reproduced.")
        if c0 == "Oracle / ceiling":
            set_text(r.cells[0].paragraphs[0], "Oracle reference")
            set_text(r.cells[1].paragraphs[0],
                     "score obtained with the ideal ratio mask; a reference point for mask models, not a strict upper bound")
        if c0.startswith("Detector macro-F1") and len(r.cells) >= 2:
            set_text(r.cells[1].paragraphs[0],
                     "≈ 0.80 across the six trained classes (vocals 0.85, drums 0.89, bass 0.88, violin 0.78, "
                     "mridangam 0.74, ghatam 0.65)")
d.save(GUIDE_OUT)
print("guide updated")

# ───────────────────────── deck ─────────────────────────
prs = Presentation(PPT)
S = list(prs.slides)


def shape(slide, name):
    return [s for s in slide.shapes if s.name == name][0]


def para_set(sh, idx, text):
    p = sh.text_frame.paragraphs[idx]
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""


s3 = S[2]
para_set(shape(s3, "TextBox 11"), 0, "Detect six trained classes; estimate guitar and piano from the Demucs backend.")
para_set(shape(s3, "TextBox 26"), 0, "Suppress unselected instruments and measure the leakage.")

s11 = S[10]
for sh in s11.shapes:
    if sh.has_chart:
        cd = CategoryChartData()
        cd.categories = ["MUSDB\nvocals", "MUSDB\ndrums", "MUSDB\nbass", "Saraga\nvocals", "Saraga\nmridangam", "Saraga\nviolin"]
        cd.add_series("Raw mixture", [-3.6, -6.5, -6.6, 0.8, -2.4, -5.9])
        cd.add_series("Our separator", [4.2, 2.3, 0.2, 5.6, 5.2, 1.1])
        sh.chart.replace_data(cd)
para_set(shape(s11, "TextBox 15"), 0,
         "Ghatam omitted: the model returns silence. Guitar and piano omitted: no separate ground truth.")

s18 = S[17]
tb = shape(s18, "TextBox 10")
para_set(tb, 1, "Violin ≈ 1 dB; absent violin and mridangam still audible")
para_set(tb, 2, "Guitar and piano: no separate ground truth, approximate output")
para_set(tb, 3, "Mask oracle reference ≈ 4–10 dB (mixture phase reused)")

s19 = S[18]
tb = shape(s19, "TextBox 9")
para_set(tb, 0, "One detector (six trained classes) and one query-conditioned separator serve eight query classes.")
para_set(tb, 2, "A competitive filter reduces leakage (SIR 11–28 dB); residual interference remains.")
prs.save(PPT)
print("deck updated")
