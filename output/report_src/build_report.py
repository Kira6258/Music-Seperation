"""
build_report.py
---------------
Builds the project report from the team's template ("Project Phase Report.docx").
The template's front matter, section layout, fonts, spacing, caption and table
styles are kept; the text, tables and figures are replaced with this project's.

Usage:
    python build_report.py <template.docx> <out.docx> [pages.json]

pages.json (optional) maps keys to page labels for the contents / figure /
table lists; the first build runs without it and a second build fills it.
"""
import copy
import json
import re
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from content_ch1_2 import CH1, CH2
from content_ch3_4 import CH3, CH4
from content_ch5_6 import CH5, CH6, REFERENCES, ABBREVIATIONS

IMG_DIR = HERE.parent / "images"

TITLE = "QUERY-CONDITIONED DEEP LEARNING FOR ADAPTIVE MULTI-SOURCE MUSIC SEPARATION"
GUIDE = "[GUIDE NAME]"
STUDENTS = [("ACHINTYA TR", "CH.SC.U4AIE24003"),
            ("BJ KIRANESH", "CH.SC.U4AIE24006"),
            ("HITHESH R", "CH.SC.U4AIE24027"),
            ("PARINITHA KANDULA", "CH.SC.U4AIE24052")]

ALL = CH1 + CH2 + CH3 + CH4 + CH5 + CH6


# ───────────────────────────── helpers ─────────────────────────────────────
def roman(n: int) -> str:
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def set_spacing(p, before=None, after=None, line=None):
    pf = p.paragraph_format
    if before is not None:
        pf.space_before = Pt(before)
    if after is not None:
        pf.space_after = Pt(after)
    if line is not None:
        pf.line_spacing = line


TOKEN = re.compile(r"([_^])\{([^}]*)\}")


def add_text_runs(p, text, size=None, bold=None, italic=None, math=False):
    """Add runs for `text`, turning _{x} / ^{x} into subscript / superscript."""
    pos = 0

    def mk(t, sub=None, sup=None):
        r = p.add_run(t)
        if size:
            r.font.size = Pt(size)
        if bold is not None:
            r.bold = bold
        if italic is not None:
            r.italic = italic
        if math:
            rpr = r._r.get_or_add_rPr()
            rf = OxmlElement("w:rFonts")
            rf.set(qn("w:ascii"), "Cambria Math")
            rf.set(qn("w:hAnsi"), "Cambria Math")
            rpr.insert(0, rf)
        if sub:
            r.font.subscript = True
        if sup:
            r.font.superscript = True
        return r

    for m in TOKEN.finditer(text):
        if m.start() > pos:
            mk(text[pos:m.start()])
        mk(m.group(2), sub=(m.group(1) == "_"), sup=(m.group(1) == "^"))
        pos = m.end()
    if pos < len(text):
        mk(text[pos:])


def clear_runs(p):
    for r in list(p._p):
        if r.tag != qn("w:pPr"):
            p._p.remove(r)


def set_par_text(p, text, keep_first_fmt=True):
    """Replace the text of a template paragraph, keeping the first run's format."""
    first = p.runs[0] if p.runs else None
    size = first.font.size if first is not None else None
    bold = first.bold if first is not None else None
    italic = first.italic if first is not None else None
    clear_runs(p)
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        r = p.add_run(ln)
        if size:
            r.font.size = size
        if bold is not None:
            r.bold = bold
        if italic is not None:
            r.italic = italic
        if i < len(lines) - 1:
            r.add_break()


def new_par(container_before, text="", align=None, before=None, after=None, line=None,
            size=None, bold=None, italic=None, math=False, keep_next=False):
    """Create a paragraph just before the final sectPr of the body."""
    p = container_before.add_paragraph()
    if align is not None:
        p.alignment = align
    set_spacing(p, before, after, line)
    if text:
        add_text_runs(p, text, size=size, bold=bold, italic=italic, math=math)
    if keep_next:
        p.paragraph_format.keep_with_next = True
    return p


def page_break(d):
    p = d.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    return p


def clone_row(tr, texts):
    new = copy.deepcopy(tr)
    tcs = new.findall(qn("w:tc"))
    for tc, t in zip(tcs, texts):
        ts = list(tc.iter(qn("w:t")))
        if ts:
            ts[0].text = t
            ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            for extra in ts[1:]:
                extra.text = ""
    return new


def rebuild_table(tbl, header_rows, proto_rows, rows_texts):
    """Keep `header_rows` rows, remove the rest and append rows cloned from
    proto_rows[kind] for every (kind, texts) in rows_texts."""
    trs = tbl._tbl.findall(qn("w:tr"))
    protos = {k: copy.deepcopy(trs[i]) for k, i in proto_rows.items()}
    for tr in trs[header_rows:]:
        tbl._tbl.remove(tr)
    for kind, texts in rows_texts:
        tbl._tbl.append(clone_row(protos[kind], texts))


# ───────────────────────── cross-reference numbering ───────────────────────
def number_blocks():
    """Walk the blocks, assign figure / table / equation numbers per chapter."""
    refs = {}
    ch = 0
    cf = ct = ce = 0
    figs, tabs, heads = [], [], []
    for b in ALL:
        k = b[0]
        if k == "chapter":
            ch = b[1]
            cf = ct = ce = 0
            heads.append(("chapter", ch, b[2]))
        elif k == "h2":
            heads.append(("h2", ch, b[1]))
        elif k == "fig":
            cf += 1
            refs["F:" + b[1]] = f"{ch}.{cf}"
            figs.append((f"Figure {ch}.{cf}", b[1]))
        elif k == "table":
            ct += 1
            refs["T:" + b[1]] = f"{ch}.{ct}"
            tabs.append((f"Table {ch}.{ct}", b[1]))
        elif k == "eq":
            ce += 1
            refs["E:" + b[1].replace("eq_", "", 1)] = f"{ch}.{ce}"
    return refs, figs, tabs, heads


REFS, FIGS, TABS, HEADS = number_blocks()


def resolve(text, number=None):
    def sub(m):
        key = m.group(1)
        if key == "N":
            return number or ""
        return REFS[key]
    return re.sub(r"\{([FTE]:[a-z0-9_]+|N)\}", sub, text)


# ───────────────────────────── main build ──────────────────────────────────
def build(template, out, pages):
    d = docx.Document(template)
    P = d.paragraphs

    def pg(key, default="0"):
        return str(pages.get(key, default))

    # --- front matter ------------------------------------------------------
    set_par_text(P[0], TITLE)

    # names on the title page: bold name, regular registration number
    p3 = P[3]
    clear_runs(p3)
    for i, (nm, reg) in enumerate(STUDENTS):
        r = p3.add_run(nm)
        r.bold = True
        r.font.size = Pt(13)
        r2 = p3.add_run(f" (Reg. No. {reg})")
        r2.bold = False
        r2.font.size = Pt(13)
        if i < len(STUDENTS) - 1:
            r2.add_break()

    set_par_text(P[5], f"Under the guidance of\nDr. {GUIDE}\nDepartment of Computer Science & Engineering")

    names_inline = ", ".join(f"{n} (Reg. No. {r})" for n, r in STUDENTS[:-1]) + f" & {STUDENTS[-1][0]} (Reg. No. {STUDENTS[-1][1]})"
    set_par_text(P[11], f"Certified that this project report entitled “{TITLE}” is the bonafide work of "
                        f"“{names_inline}”, who carried out the project work under my supervision.")
    set_par_text(P[12], "SIGNATURE\tSIGNATURE")
    set_par_text(P[13], f"Dr. S. BAGHAVATHI PRIYA\tDr. {GUIDE}")
    set_par_text(P[15], "Department of CSE\tDept. of CSE")

    set_par_text(P[22], f"We declare that the report entitled “{TITLE}” submitted by us for the degree of "
                        f"Bachelor of Technology in Computer Science and Engineering is the record of the project work "
                        f"carried out by us under the guidance of “Dr. {GUIDE}” and this work has not formed the "
                        f"basis for the award of any degree, diploma, associateship, fellowship, titled in this or any other "
                        f"University or other similar institution of higher learning.")
    sig = "\n".join(f"{n} ({r})" for n, r in STUDENTS)
    set_par_text(P[23], "SIGNATURE OF THE CANDIDATES\n\n\n" + sig +
                 "\n\nDepartment of Computer Science & Engineering (AIE)\nAmrita School of Computing, Chennai Campus.")

    abstract = [
        "Most music source separation systems return a fixed list of stems, usually vocals, drums, bass and a catch-all "
        "\"other\". A listener who wants a violin, a mridangam or a ghatam gets nothing from them, and a listener who selects "
        "one instrument often still hears the others. Indian classical instruments are the clearest example, because the "
        "pretrained separators were built and trained on Western music.",
        "This project builds a system that detects which instruments are present in a recording and returns the stems the "
        "user selects while suppressing the others. A small convolutional detector is trained for six classes (vocals, "
        "drums, bass, violin, mridangam, ghatam) and reaches a macro-F1 of 0.798 across them. Guitar and piano presence is "
        "estimated from the energy outputs of a pretrained 6-source Demucs model, because neither training corpus labels "
        "them separately. A single query-conditioned U-Net (8.8 million parameters, FiLM at four encoder levels) is trained "
        "on MUSDB18-HQ and Saraga, with 20% of its requests naming stems that are absent. A competitive mask then divides "
        "every time-frequency bin among the Demucs and U-Net sources so that unselected stems are attenuated.",
        "Repairing faults found in the data raised best validation SI-SDR from -8.48 dB to +2.53 dB. On 12 Carnatic "
        "validation tracks the trained U-Net scored 0.8, 2.8 and 4.4 dB above the best-matching Demucs output for vocals, "
        "violin and mridangam. On four MUSDB test tracks the final system reached an SIR of 28.2 dB for vocals and 27.6 dB "
        "for drums, with residual leakage that is largest for bass at 11.0 dB. Ghatam is not recovered, violin is modest at "
        "about 1 dB, requests for absent violin and mridangam still return audible output, and performance on real, "
        "unsummed Carnatic mixes is not measured. A web application accepts audio and video files.",
    ]
    for i, t in zip((26, 27, 28), abstract):
        set_par_text(P[i], t)
    kw = P[29]
    kw.runs[1].text = ("Music Source Separation, Query-Conditioned Separation, FiLM, U-Net, Stem Detection, Carnatic Music, "
                       "Mridangam, Competitive Masking, SI-SDR, Absent-Query Training.")

    ack = [
        "This project work would not have been possible without the contribution of many people. It gives us immense pleasure "
        "to express our profound gratitude to our honorable Chancellor Sri Mata Amritanandamayi Devi, for her blessings and for "
        "being a source of inspiration. We are indebted to extend our gratitude to our Administrative Director, Swami "
        "Vinayamritananda Puri and Campus Director, Mr. I B Manikantan, Amrita School of Computing and Engineering, for "
        "facilitating us all the facilities and extended support to gain valuable education and learning experience.",
        f"We register our special thanks to Dr. V. Jayakumar, Principal, Amrita School of Computing and Engineering for the "
        f"support given to us in the successful conduct of this project. We wish to express our sincere gratitude to "
        f"Dr. S. Baghavathi Priya, Chairperson, Dr. Annamalai R, Program Chair, and to our supervisor Dr. {GUIDE}, "
        f"Department of Computer Science and Engineering, for their inspiring guidance, personal involvement, and constant "
        f"encouragement during the entire course of this work.",
        "We are grateful to the Project Coordinator, Review Panel Members, and the entire faculty of the Department of "
        "Computer Science & Engineering, for their constructive criticisms and valuable suggestions which have been a rich "
        "source to improve the quality of this work.",
    ]
    for i, t in zip((32, 33, 34), ack):
        set_par_text(P[i], t)
    set_par_text(P[35], sig + "\n\nDepartment of Computer Science & Engineering (AIE)\nAmrita School of Computing, Chennai Campus.")

    # --- list tables -------------------------------------------------------
    T = d.tables
    toc_rows = [("front", ["Abstract", pg("ABSTRACT")]),
                ("front", ["List of Figures", pg("LIST OF FIGURES")]),
                ("front", ["List of Tables", pg("LIST OF TABLES")]),
                ("front", ["List of Symbols and Abbreviations", pg("LIST OF SYMBOLS AND ABBREVIATIONS")])]
    for kind, ch, text in HEADS:
        if kind == "chapter":
            title = text.replace("&", "and") if False else text
            toc_rows.append(("front", [f"CHAPTER {ch}   {title}", pg(f"H:{ch}")]))
        else:
            m = re.match(r"(\d+\.\d+)\s+(.*)", text)
            toc_rows.append(("sec", [f"    {m.group(1)}  {m.group(2)}", pg(f"H:{text}")]))
    toc_rows.append(("front", ["REFERENCES", pg("REFERENCES")]))
    trs = T[0]._tbl.findall(qn("w:tr"))
    rebuild_table(T[0], 1, {"front": 1, "sec": 6}, toc_rows)

    fig_rows = []
    for num, key in FIGS:
        cap = next(b for b in ALL if b[0] == "fig" and b[1] == key)[4]
        title = re.sub(r"^Figure \{N\}:\s*", "", cap)
        fig_rows.append(("r", [num, title, pg(f"FIG:{num}")]))
    rebuild_table(T[1], 1, {"r": 1}, fig_rows)

    tab_rows = []
    for num, key in TABS:
        cap = next(b for b in ALL if b[0] == "table" and b[1] == key)[2]
        title = re.sub(r"^Table \{N\}:\s*", "", cap)
        tab_rows.append(("r", [num, title, pg(f"TAB:{num}")]))
    rebuild_table(T[2], 1, {"r": 1}, tab_rows)

    # symbols table: reuse row 0 as prototype
    rebuild_table(T[3], 0, {"r": 0}, [("r", [a, "-   " + b]) for a, b in ABBREVIATIONS])

    P[47].paragraph_format.page_break_before = True
    # tighten the spacer paragraphs between list of figures and list of tables
    for i in (43, 44, 45, 46):
        P[i]._p.getparent().remove(P[i]._p)

    # --- drop the template's body (everything after the section break) -----
    body = d.element.body
    children = list(body.iterchildren())
    start = children.index(P[52]._p)
    for el in children[start:]:
        if el.tag == qn("w:sectPr"):
            continue
        body.remove(el)

    # remove picture parts that are no longer referenced
    xml = d.element.xml
    part = d.part
    for rid, rel in list(part.rels.items()):
        if rel.reltype.endswith("/image") and f'"{rid}"' not in xml:
            part.rels.pop(rid)

    # --- body content ------------------------------------------------------
    eq_no = {}
    cur_ch = 0
    ce = 0
    first_chapter = True
    for b in ALL:
        k = b[0]
        if k == "chapter":
            cur_ch = b[1]
            ce = 0
            if not first_chapter:
                page_break(d)
            first_chapter = False
            new_par(d, f"CHAPTER {cur_ch}", WD_ALIGN_PARAGRAPH.CENTER, 24, 6, 1.0, size=16, bold=True)
            new_par(d, b[2], WD_ALIGN_PARAGRAPH.CENTER, None, 18, 1.0, size=16, bold=True, keep_next=True)
        elif k == "intro":
            new_par(d, resolve(b[1]), WD_ALIGN_PARAGRAPH.JUSTIFY, None, None, 1.5)
        elif k == "h2":
            new_par(d, b[1], None, 14, 6, None, size=14, bold=True, keep_next=True)
        elif k == "h3":
            new_par(d, b[1], None, 10, 4, None, bold=True, keep_next=True)
        elif k == "p":
            new_par(d, resolve(b[1]), WD_ALIGN_PARAGRAPH.JUSTIFY, None, 6, 1.5)
        elif k == "li":
            new_par(d, f"{b[1]}. {b[2]}: {resolve(b[3])}", WD_ALIGN_PARAGRAPH.JUSTIFY, 2, 4, 1.5)
        elif k == "eq":
            ce += 1
            p = d.add_paragraph()
            set_spacing(p, 8, 8)
            ppr = p._p.get_or_add_pPr()
            tabs = OxmlElement("w:tabs")
            for val, pos in (("center", 4500), ("right", 9000)):
                t = OxmlElement("w:tab")
                t.set(qn("w:val"), val)
                t.set(qn("w:pos"), str(pos))
                tabs.append(t)
            ppr.insert(0, tabs)
            p.add_run().add_tab()
            add_text_runs(p, resolve(b[2]), size=11.5, math=True)
            p.add_run().add_tab()
            r = p.add_run(f"({cur_ch}.{ce})")
            r.bold = True
            r.font.size = Pt(11)
        elif k == "fig":
            num = REFS["F:" + b[1]]
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_spacing(p, 10, 4)
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(IMG_DIR / b[2]), width=Cm(b[3]))
            cap = d.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_spacing(cap, 2, 12)
            r = cap.add_run(resolve(b[4], num))
            r.bold = True
            r.italic = True
            r.font.size = Pt(10.5)
        elif k == "table":
            num = REFS["T:" + b[1]]
            cap = d.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_spacing(cap, 10, 4)
            cap.paragraph_format.keep_with_next = True
            r = cap.add_run(resolve(b[2], num))
            r.bold = True
            r.font.size = Pt(11)
            add_table(d, b[3], b[4], b[5])
            sp = d.add_paragraph()
            set_spacing(sp, None, 6)
        elif k == "refs":
            pass

    # --- references --------------------------------------------------------
    page_break(d)
    new_par(d, "REFERENCES", WD_ALIGN_PARAGRAPH.CENTER, 18, 18, 1.5, size=16, bold=True)
    for i, t in enumerate(REFERENCES, 1):
        new_par(d, f"[{i}] {t}", WD_ALIGN_PARAGRAPH.JUSTIFY, 3, 6, 1.25, size=10.5)

    d.save(out)


def add_table(d, header, rows, widths):
    n = len(header)
    t = d.add_table(rows=1 + len(rows), cols=n)
    tbl = t._tbl
    # fixed layout, centred, as in the template
    tblPr = tbl.tblPr
    for el in list(tblPr):
        tblPr.remove(el)
    w = OxmlElement("w:tblW")
    w.set(qn("w:w"), "0")
    w.set(qn("w:type"), "auto")
    tblPr.append(w)
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    tblPr.append(jc)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    look = OxmlElement("w:tblLook")
    look.set(qn("w:val"), "04A0")
    tblPr.append(look)
    grid = tbl.find(qn("w:tblGrid"))
    for gc, wd in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(wd))

    def border(tc, top=None, bottom=None):
        tcPr = tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        for side, sz in (("top", top), ("bottom", bottom)):
            e = OxmlElement(f"w:{side}")
            if sz:
                e.set(qn("w:val"), "single")
                e.set(qn("w:sz"), str(sz))
            else:
                e.set(qn("w:val"), "none")
                e.set(qn("w:sz"), "0")
            e.set(qn("w:space"), "0")
            e.set(qn("w:color"), "000000")
            b.append(e)
        tcPr.append(b)

    def fill(cell, wd, text, header_cell, left, top, bottom, keep_next=True):
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        for el in list(tcPr):
            if el.tag == qn("w:tcW"):
                tcPr.remove(el)
        tcw = OxmlElement("w:tcW")
        tcw.set(qn("w:w"), str(wd))
        tcw.set(qn("w:type"), "dxa")
        tcPr.insert(0, tcw)
        border(tc, top, bottom)
        if header_cell:
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "F1F5F9")
            tcPr.append(shd)
        p = cell.paragraphs[0]
        set_spacing(p, 4 if header_cell else 3, 4 if header_cell else 3)
        p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if header_cell
                       else (WD_ALIGN_PARAGRAPH.LEFT if left else WD_ALIGN_PARAGRAPH.CENTER))
        add_text_runs(p, resolve(text), size=10.5 if header_cell else 10, bold=True if header_cell else None)
        if keep_next:
            p.paragraph_format.keep_with_next = True

    for ci, (h, wd) in enumerate(zip(header, widths)):
        fill(t.rows[0].cells[ci], wd, h, True, False, 12, 6)
    for ri, row in enumerate(rows, 1):
        last = ri == len(rows)
        for ci, (v, wd) in enumerate(zip(row, widths)):
            left = ci == 0 or len(v) > 38
            fill(t.rows[ri].cells[ci], wd, v, False, left, 6, 12 if last else None, keep_next=not last)
    # center rows
    for tr in tbl.findall(qn("w:tr")):
        trPr = tr.get_or_add_trPr()
        j = OxmlElement("w:jc")
        j.set(qn("w:val"), "center")
        trPr.append(j)
        cs = OxmlElement("w:cantSplit")
        trPr.insert(0, cs)
    return t


if __name__ == "__main__":
    tpl, out = sys.argv[1], sys.argv[2]
    pages = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")) if len(sys.argv) > 3 else {}
    build(tpl, out, pages)
    # emit the list of strings whose page numbers are needed
    need = [("ABSTRACT", "ABSTRACT"), ("LIST OF FIGURES", "LIST OF FIGURES"),
            ("LIST OF TABLES", "LIST OF TABLES"),
            ("LIST OF SYMBOLS AND ABBREVIATIONS", "LIST OF SYMBOLS AND ABBREVIATIONS")]
    body_items = []
    for kind, ch, text in HEADS:
        if kind == "chapter":
            body_items.append((f"H:{ch}", f"CHAPTER {ch}"))
        else:
            body_items.append((f"H:{text}", text))
    # interleave figures / tables in document order
    order = []
    ch = 0
    cf = ct = 0
    for b in ALL:
        if b[0] == "chapter":
            ch, cf, ct = b[1], 0, 0
            order.append((f"H:{ch}", f"CHAPTER {ch}"))
        elif b[0] == "h2":
            order.append((f"H:{b[1]}", b[1]))
        elif b[0] == "fig":
            cf += 1
            cap = resolve(b[4], f"{ch}.{cf}")
            order.append((f"FIG:Figure {ch}.{cf}", cap))
        elif b[0] == "table":
            ct += 1
            cap = resolve(b[2], f"{ch}.{ct}")
            order.append((f"TAB:Table {ch}.{ct}", cap))
    order.append(("REFERENCES", "REFERENCES"))
    Path(out).with_suffix(".need.json").write_text(
        json.dumps({"front": need, "body": order}, ensure_ascii=False), encoding="utf-8")
    print("built", out, "| figures", len(FIGS), "| tables", len(TABS))
