"""Updates Final_Review_Music_Separation.pptx: video wording + one slide of real example outputs."""
import copy
import sys

from pptx import Presentation
from pptx.util import Inches, Emu
from PIL import Image

SRC = sys.argv[1]
IMG = sys.argv[2]
prs = Presentation(SRC)
slides = list(prs.slides)

# ---- slide 15 (web application): mention video input -------------------------
web = slides[14]
for sh in web.shapes:
    if not sh.has_text_frame:
        continue
    if sh.text_frame.text.strip() == "Upload a song":
        sh.text_frame.paragraphs[0].runs[0].text = "Upload audio or a video file"
        for r in sh.text_frame.paragraphs[0].runs[1:]:
            r.text = ""
    if sh.text_frame.text.strip() == "A/B compare with the original":
        sh.text_frame.paragraphs[0].runs[0].text = "A/B compare, or rebuild the video"
        for r in sh.text_frame.paragraphs[0].runs[1:]:
            r.text = ""

# ---- new slide: real example outputs (after the leakage slide, index 14) -----
src = slides[13]                       # "Suppressing Unselected Stems"
blank = src.slide_layout
new = prs.slides.add_slide(blank)
for sh in list(new.shapes):
    sh._element.getparent().remove(sh._element)

by_name = {sh.name: sh for sh in src.shapes}
for nm in ("Group 2", "Group 4", "Group 6", "Slide Title", "Rounded Rectangle 8",
           "Rounded Rectangle 11", "TextBox 12", "Rounded Rectangle 13", "TextBox 14", "TextBox 15"):
    el = copy.deepcopy(by_name[nm]._element)
    # pictures inside copied groups point at the SOURCE slide's relationships;
    # re-create those relationships on the new slide and re-point the blips
    for blip in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
        key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
        old = blip.get(key)
        if old:
            part = src.part.related_part(old)
            blip.set(key, new.part.relate_to(part, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"))
    new.shapes._spTree.append(el)

shapes = {sh.name: sh for sh in new.shapes}


def set_first_run(shape, texts):
    """texts: one string per paragraph; keeps the first run's formatting."""
    paras = shape.text_frame.paragraphs
    for p, t in zip(paras, texts):
        runs = p.runs
        runs[0].text = t
        for r in runs[1:]:
            r.text = ""


set_first_run(shapes["Slide Title"], ["Example Outputs"])

# picture inside the big card
w_img, h_img = Image.open(IMG).size
card = shapes["Rounded Rectangle 8"]
max_h = Inches(6.5)
max_w = card.width - Inches(0.4)
scale = min(max_w / w_img, max_h / h_img)
pw, ph = int(w_img * scale), int(h_img * scale)
left = card.left + (card.width - pw) // 2
top = card.top + (card.height - ph) // 2
new.shapes.add_picture(IMG, left, top, width=pw, height=ph)

# right column: two number cards (reused) + a third (cloned) + note
c1, t1 = shapes["Rounded Rectangle 11"], shapes["TextBox 12"]
c2, t2 = shapes["Rounded Rectangle 13"], shapes["TextBox 14"]
note = shapes["TextBox 15"]

top0 = Inches(2.70)
card_h = Inches(1.95)
gap = Inches(0.2)
c3_el = copy.deepcopy(c2._element)
t3_el = copy.deepcopy(t2._element)
new.shapes._spTree.append(c3_el)
new.shapes._spTree.append(t3_el)
c3 = [s for s in new.shapes if s._element is c3_el][0]
t3 = [s for s in new.shapes if s._element is t3_el][0]

for i, (c, t) in enumerate(((c1, t1), (c2, t2), (c3, t3))):
    c.top = top0 + i * (card_h + gap)
    c.height = card_h
    t.top = c.top + Inches(0.12)
    t.height = card_h - Inches(0.2)

set_first_run(t1, ["+11.7 dB", "vocals, MUSDB test excerpt"])
set_first_run(t2, ["+4.3 dB", "drums, MUSDB test excerpt"])
set_first_run(t3, ["+11.2 dB", "mridangam, Saraga validation clip"])

note.top = top0 + 3 * (card_h + gap)
note.height = Inches(1.2)
set_first_run(note, ["Real clips, not chosen for quality. SI-SDR is against the true stem. The mridangam clip is far above the 5.2 dB validation average."])
nm = note.text_frame
for p in nm.paragraphs[1:]:
    for r in p.runs:
        r.text = ""

# move the new slide to position 14 (0-based) = right after the leakage slide
lst = prs.slides._sldIdLst
ids = list(lst)
el = ids[-1]
lst.remove(el)
lst.insert(14, el)

# speaker notes
new.notes_slide.notes_text_frame.text = (
    "Three real clips from the final system. Vocals and drums come from the pretrained backend through the competitive "
    "mask, and mridangam comes from our trained U-Net. The clips were taken as they came: the second MUSDB test track "
    "at 60 to 72 seconds, and the first Saraga validation clip that has audible mridangam. They are single examples, "
    "and the mridangam clip scores well above the validation average of 5.2 dB, so quote the averages from the earlier "
    "slides, not these numbers.")

prs.save(SRC)
print("slides:", len(prs.slides))
