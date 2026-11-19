# -*- coding: utf-8 -*-
"""Общие стили и помощники reportlab для конспекта (кириллица через DejaVu)."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted,
                                Table, TableStyle)

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONT_DIR / "DejaVuSansMono.ttf")))
registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                   italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")

NAVY = colors.HexColor("#1f3b57")
BLUE = colors.HexColor("#2f5d86")
_ss = getSampleStyleSheet()

TITLE = ParagraphStyle("T", parent=_ss["Title"], fontName="DejaVu-Bold", fontSize=22,
                       leading=28, alignment=TA_CENTER, spaceAfter=6)
SUBTITLE = ParagraphStyle("ST", parent=_ss["Normal"], fontName="DejaVu", fontSize=13,
                          leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
PART = ParagraphStyle("PART", parent=_ss["Heading1"], fontName="DejaVu-Bold", fontSize=18,
                      leading=24, spaceBefore=4, spaceAfter=10, textColor=colors.white,
                      backColor=NAVY, borderPadding=(6, 6, 6, 6))
H1 = ParagraphStyle("H1", parent=_ss["Heading1"], fontName="DejaVu-Bold", fontSize=14.5, keepWithNext=1,
                    leading=19, spaceBefore=10, spaceAfter=5, textColor=NAVY)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontName="DejaVu-Bold", fontSize=11.5, keepWithNext=1,
                    leading=15, spaceBefore=7, spaceAfter=3, textColor=BLUE)
BODY = ParagraphStyle("B", parent=_ss["BodyText"], fontName="DejaVu", fontSize=9.8,
                      leading=14.2, alignment=TA_JUSTIFY, spaceAfter=4)
BULLET = ParagraphStyle("BL", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2)
CAPTION = ParagraphStyle("C", parent=_ss["Normal"], fontName="DejaVu-Oblique", fontSize=8.5,
                         leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
                         spaceBefore=2, spaceAfter=8)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.5, leading=12,
                       textColor=colors.HexColor("#555555"))
CENTER = ParagraphStyle("CE", parent=BODY, alignment=TA_CENTER)
FORMULA = ParagraphStyle("F", parent=BODY, fontName="DejaVu", alignment=TA_CENTER,
                         fontSize=10.5, leading=15, backColor=colors.HexColor("#f4f6f9"),
                         borderPadding=(4, 4, 4, 4), spaceBefore=4, spaceAfter=7)
NOTE = ParagraphStyle("N", parent=BODY, backColor=colors.HexColor("#fff4d6"),
                      borderColor=colors.HexColor("#e0b84c"), borderWidth=0.6,
                      borderPadding=(5, 5, 5, 5), spaceBefore=4, spaceAfter=8)
KEY = ParagraphStyle("K", parent=BODY, backColor=colors.HexColor("#e7f1e3"),
                     borderColor=colors.HexColor("#6aa84f"), borderWidth=0.6,
                     borderPadding=(5, 5, 5, 5), spaceBefore=4, spaceAfter=8)
Q = ParagraphStyle("Q", parent=BODY, fontName="DejaVu-Bold", textColor=NAVY,
                   spaceBefore=5, spaceAfter=1)
A = ParagraphStyle("A", parent=BODY, leftIndent=10)
CODE = ParagraphStyle("CODE", fontName="DejaVuMono", fontSize=7.9, leading=10.2,
                      backColor=colors.HexColor("#f3f3f3"), borderColor=colors.HexColor("#cccccc"),
                      borderWidth=0.5, borderPadding=(4, 4, 4, 4), spaceBefore=3, spaceAfter=8,
                      leftIndent=2)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullet(text):
    return Paragraph(text, BULLET, bulletText="•")


def num(text, n):
    return Paragraph(text, BULLET, bulletText=f"{n}.")


def note(text):
    return Paragraph("<b>Важно.</b> " + text, NOTE)


def key(text):
    return Paragraph("<b>Запомнить.</b> " + text, KEY)


def formula(text):
    return Paragraph(text, FORMULA)


def code(text):
    return Preformatted(text.strip("\n"), CODE)


def qa(q, a):
    return KeepTogether([Paragraph("В: " + q, Q), Paragraph("О: " + a, A)])


def image(path, max_width=170 * mm, max_height=110 * mm):
    path = Path(path)
    with PILImage.open(path) as im:
        w, h = im.size
    r = h / w
    width = max_width
    height = width * r
    if height > max_height:
        height = max_height
        width = height / r
    return Image(str(path), width=width, height=height)


def figure(path, caption, max_width=170 * mm, max_height=110 * mm):
    return KeepTogether([image(path, max_width, max_height), p(caption, CAPTION)])


def table(data, col_widths=None, font_size=8.5, header=True, first_left=True):
    data = [[c if not isinstance(c, str) else Paragraph(c, ParagraphStyle(
        "cell", fontName="DejaVu-Bold" if (header and i == 0) else "DejaVu",
        fontSize=font_size, leading=font_size * 1.3,
        textColor=colors.white if (header and i == 0) else colors.black))
        for c in row] for i, row in enumerate(data)]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a9b6c2")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23445f")),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                  [colors.white, colors.HexColor("#eef3f8")])]
    t.setStyle(TableStyle(cmds))
    return t


def esc(s):
    return escape(str(s))
