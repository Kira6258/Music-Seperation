"""Adds the report-figure / report-build section to Codebase_Guide_for_Teammates.docx."""
import copy
import sys

import docx
from docx.oxml.ns import qn

SRC = sys.argv[1]
d = docx.Document(SRC)
P = d.paragraphs

# prototypes (copied from existing, already-styled elements)
h2_proto = P[561]._p            # "9.2 diagnose_separator.py"
body_proto = P[562]._p          # "Purpose. ..." paragraph (bold lead-in + normal text)
plain_proto = P[566]._p         # plain explanatory paragraph
h3_proto = P[563]._p            # "main" sub-heading
anchor = P[601]._p              # "10. The inference pipeline" heading: insert before it


def set_runs(p_el, parts):
    """parts: list of (text, bold). Reuse first run formats loosely."""
    runs = p_el.findall(qn("w:r"))
    proto_runs = [copy.deepcopy(r) for r in runs]
    for r in runs:
        p_el.remove(r)
    for text, bold in parts:
        base = copy.deepcopy(proto_runs[0] if bold else proto_runs[-1])
        for child in list(base):
            if child.tag != qn("w:rPr"):
                base.remove(child)
        rpr = base.find(qn("w:rPr"))
        if rpr is not None:
            for tag in ("w:rFonts",):
                for e in rpr.findall(qn(tag)):
                    rpr.remove(e)
            b = rpr.find(qn("w:b"))
            if bold and b is None:
                rpr.insert(0, rpr.makeelement(qn("w:b"), {}))
            if not bold and b is not None:
                rpr.remove(b)
        t = base.makeelement(qn("w:t"), {})
        t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        base.append(t)
        p_el.append(base)


def clone_par(proto, parts):
    el = copy.deepcopy(proto)
    set_runs(el, parts)
    return el


new = []
h = copy.deepcopy(h2_proto)
set_runs(h, [("9.4 Report figures and the report build (output/)", False)])
new.append(h)
new.append(clone_par(body_proto, [("Purpose. ", True),
    ("These scripts produce the figures and the written project report. They do not touch the models, the checkpoints or the "
     "web application, and nothing in training or inference imports them. The files live in output/; the generated pictures "
     "are written to output/images/.", False)]))

rows = [
    ("output/make_figures.py",
     "Draws the diagrams (U-Net with FiLM, competitive mask, system architecture, dataset composition, detector) and the "
     "result charts (validation progress, per-stem SI-SDR, detector F1, comparison with Demucs, SIR). The chart values are "
     "the measured numbers listed in Section 14. Run: python output/make_figures.py"),
    ("output/make_example_figure.py",
     "Runs the real inference pipeline on one MUSDB test excerpt and one Saraga validation clip and draws mixture, output "
     "and ground-truth spectrograms with the SI-SDR of each output. Needs the checkpoints and the datasets. "
     "Run: python output/make_example_figure.py"),
    ("output/report_src/",
     "build_report.py fills the team's report template (fonts, spacing, caption and table styles are kept) from the "
     "content_ch*.py files, which hold the text of each chapter. Run: python build_report.py <template.docx> <out.docx> "
     "[pages.json]. The contents, figure and table lists show page numbers taken from pages.json, which is produced by "
     "rendering the first build in Word."),
    ("output/images/",
     "All generated figures, including a screenshot of the web interface (fig_4_4_web_interface.png) taken from the "
     "running application."),
]

# build the small table by copying an existing two-column table (Parameter / Meaning)
tbl = copy.deepcopy(d.tables[8]._tbl)
trs = tbl.findall(qn("w:tr"))
header, proto = trs[0], trs[1]
for tr in trs[1:]:
    tbl.remove(tr)
for tc, txt in zip(header.findall(qn("w:tc")), ("File", "What it does")):
    ts = list(tc.iter(qn("w:t")))
    ts[0].text = txt
    for e in ts[1:]:
        e.text = ""
for a, b in rows:
    tr = copy.deepcopy(proto)
    for tc, txt in zip(tr.findall(qn("w:tc")), (a, b)):
        ts = list(tc.iter(qn("w:t")))
        ts[0].text = txt
        for e in ts[1:]:
            e.text = ""
    tbl.append(tr)
new.append(tbl)
new.append(copy.deepcopy(P[565]._p))   # empty spacer paragraph
new.append(clone_par(plain_proto, [
    ("The numbers in the report come from the diagnostic and evaluation scripts described in Sections 9.1 and 9.2. If a "
     "checkpoint or a result changes, update the values at the top of make_figures.py and the numbers in the chapter "
     "text before rebuilding. The figure of example outputs is generated from the live models, so it changes by itself "
     "when the checkpoint changes.", False)]))
new.append(copy.deepcopy(P[565]._p))

for el in new:
    anchor.addprevious(el)

# quick-start table (Section 1): one more row
t0 = d.tables[0]
last = t0.rows[-1]._tr
nr = copy.deepcopy(last)
for tc, txt in zip(nr.findall(qn("w:tc")), ("regenerate the report figures or rebuild the report", "Section 9.4 (output/)")):
    ts = list(tc.iter(qn("w:t")))
    ts[0].text = txt
    for e in ts[1:]:
        e.text = ""
last.addnext(nr)

# file map (Section 4): add a row before the datasets/ row
t5 = d.tables[5]
proto_row = t5.rows[-2]._tr            # outputs/ row
nr = copy.deepcopy(proto_row)
for tc, txt in zip(nr.findall(qn("w:tc")),
                   ("output/", "Report material", "make_figures.py, make_example_figure.py, report_src/, images/")):
    ts = list(tc.iter(qn("w:t")))
    ts[0].text = txt
    for e in ts[1:]:
        e.text = ""
proto_row.addnext(nr)

d.save(SRC)
print("guide updated")
