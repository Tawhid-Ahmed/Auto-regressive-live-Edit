"""Parse lightweight math markup and render Word runs with sub/superscript."""
from __future__ import annotations

from typing import Literal

from docx.text.paragraph import Paragraph
from docx.shared import Pt

Style = Literal["normal", "bold", "sub", "sup"]
MATH_FONT = "Cambria Math"

_SUB_SUP_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789<>=∈−+.,|ℓθσλΣΔ"
)


def tokenize_math(text: str) -> list[tuple[str, Style]]:
    """Tokenize **bold**, _{sub}, _x, ^{sup}, ^(sup), ^word."""
    tokens: list[tuple[str, Style]] = []
    i = 0
    n = len(text)

    def read_braced(open_ch: str, close_ch: str, start: int) -> tuple[str, int] | None:
        if start >= n or text[start] != open_ch:
            return None
        j = start + 1
        while j < n:
            if text[j] == close_ch:
                return text[start + 1 : j], j + 1
            j += 1
        return None

    def read_token(start: int) -> tuple[str, int]:
        j = start
        while j < n and text[j] in _SUB_SUP_CHARS:
            j += 1
        return text[start:j], j

    while i < n:
        if text.startswith("**", i):
            j = text.find("**", i + 2)
            if j != -1:
                tokens.append((text[i + 2 : j], "bold"))
                i = j + 2
                continue

        if text[i] == "^":
            i += 1
            if i < n and text[i] == "{":
                parsed = read_braced("{", "}", i)
                if parsed:
                    inner, i = parsed
                    tokens.append((inner, "sup"))
                    continue
            if i < n and text[i] == "(":
                parsed = read_braced("(", ")", i)
                if parsed:
                    inner, i = parsed
                    tokens.append((inner, "sup"))
                    continue
            tok, i = read_token(i)
            if tok:
                tokens.append((tok, "sup"))
                continue
            if i < n:
                tokens.append((text[i], "sup"))
                i += 1
            continue

        if text[i] == "_":
            i += 1
            if i < n and text[i] == "{":
                parsed = read_braced("{", "}", i)
                if parsed:
                    inner, i = parsed
                    tokens.append((inner, "sub"))
                    continue
            tok, i = read_token(i)
            if tok:
                tokens.append((tok, "sub"))
                continue
            if i < n:
                tokens.append((text[i], "sub"))
                i += 1
            continue

        j = i
        while j < n and not text.startswith("**", j) and text[j] not in "^_":
            j += 1
        chunk = text[i:j]
        if chunk:
            tokens.append((chunk, "normal"))
        i = j

    return tokens


def add_formatted_text(
    paragraph: Paragraph,
    text: str,
    *,
    size: int = 11,
    font: str = MATH_FONT,
    equation: bool = False,
) -> None:
    """Append runs with native Word sub/superscript where markup appears."""
    eq_size = 12 if equation else size
    sub_sup_size = max(eq_size - 3, 8)

    for segment, style in tokenize_math(text):
        if not segment:
            continue
        run = paragraph.add_run(segment)
        run.font.name = font
        if style == "bold":
            run.bold = True
            run.font.size = Pt(eq_size)
        elif style == "sub":
            run.font.subscript = True
            run.font.size = Pt(sub_sup_size)
        elif style == "sup":
            run.font.superscript = True
            run.font.size = Pt(sub_sup_size)
        else:
            run.font.size = Pt(eq_size)


def fill_table_cell(cell, text: str, *, bold: bool = False, size: int = 10) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    add_formatted_text(p, text.replace("**", ""), size=size, font=MATH_FONT)
    if bold:
        for run in p.runs:
            run.bold = True
