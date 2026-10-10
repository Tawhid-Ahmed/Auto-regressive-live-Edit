"""One-shot export: review status -> xlsx; details/summary -> docx."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

OUT_DIR = Path(__file__).resolve().parents[1] / "doc"
XLSX_PATH = OUT_DIR / "AAAI_TO_CVPR_SUPERVISOR_PROGRESS_STATUS.xlsx"
DOCX_PATH = OUT_DIR / "AAAI_TO_CVPR_SUPERVISOR_PROGRESS_DETAILS.docx"

# (num, point, status, writing_only, whats_left)
R1 = [
    (1, "Own edit on wrong image not counted as error", "Done", "No", "—"),
    (2, "Partner edit on matching image wrongly counted as error", "Done", "No", "—"),
    (3, "Swap-accuracy direction conflicted with safety goal", "Measured; no win yet", "No", "Improve answer-level (cut stale answers on swap); then we can claim it"),
    (4, "Components not isolated (vision / text / LiveEdit / fusion)", "Done", "No", "—"),
    (5, "Binary false-activation may overstate baseline leak", "Done", "No", "—"),
    (6, "Only 100 edits; pool size untested", "Partly done", "No", "Optional: more edit orders + Easy-VQA pool-size curve"),
    (7, "Repeated prompts / visual collisions may explain residual", "Partly done", "No", "Optional: hard-neg difficulty strata + small human check of pairs"),
    (8, "Only one backbone and one editor", "In progress", "No", "Finish LLaVA Dual-Expert Keys train + eval; then other editors"),
    (9, "Residual leak still high", "Measured; no win yet", "No", "Method retry to cut absolute hard-neg leak under honest pools"),
]
R2 = [
    (1, "CARML / MSCKE not compared", "Partly done", "No", "MSCKE public-code check + related-work write-up"),
    (2, "Leak metric may be LiveEdit-specific", "Not started", "No", "Run same leak protocol on other editors"),
    (3, "Only author-built baseline", "Done", "No", "—"),
    (4, "No gain on Rel / Gen / Loc", "Partly done", "Yes", "Write paper section / plot leading with leak vs reliability"),
    (5, "Single BLIP-2 backbone", "In progress", "No", "Finish LLaVA Dual-Expert Keys train + eval (due 15 Oct)"),
]
AI = [
    (1, "Conflicting targets (FPR / Swap / equation)", "Partly done", "No", "Answer-level still weak; do not claim swap-answer rescue until fixed"),
    (2, "Missing always-reject / routing controls", "Partly done", "Yes", "Paper polish: present controls clearly in tables/text"),
    (3, "Novelty vs LiveEdit / CARML unclear", "Partly done", "Yes", "Write side-by-side routing-spec table in the paper"),
    (4, "No Dual-Expert Keys without adversarial training; no vision/text-only", "Done", "No", "—"),
    (5, "Lifelong scaling claimed but not tested", "Done", "No", "—"),
    (6, "Prototype score underspecified", "Not started", "Yes", "Write concrete prototype-score definition in method section"),
    (7, "Hard-negative difficulty strata + human check", "Later", "No", "Optional later if time remains"),
    (8, "Explain VLKEB vs Easy-VQA residual gap", "Done", "No", "—"),
    (9, "Multi-seed reuses same checkpoints", "Not started", "No", "Run five evaluation seeds (and training seeds if time)"),
    (10, "Hard-negative missing from VLKEB main tables", "Done", "No", "—"),
    (11, "Writing nits (terms, spelling, hyperparameters)", "Partly done", "Yes", "Clean draft: spelling, terms, figure seed labels, report hyperparameters"),
]


def write_xlsx() -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Review progress"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="1F4E79")
    group_fill = PatternFill("solid", fgColor="D6E3F0")
    group_font = Font(bold=True, size=12)
    thin = Border(
        left=Side(style="thin", color="B0B0B0"),
        right=Side(style="thin", color="B0B0B0"),
        top=Side(style="thin", color="B0B0B0"),
        bottom=Side(style="thin", color="B0B0B0"),
    )
    status_fills = {
        "Done": PatternFill("solid", fgColor="C6EFCE"),
        "Partly done": PatternFill("solid", fgColor="FFE699"),
        "In progress": PatternFill("solid", fgColor="BDD7EE"),
        "Not started": PatternFill("solid", fgColor="F8CBAD"),
        "Later": PatternFill("solid", fgColor="E2D5F1"),
        "Measured; no win yet": PatternFill("solid", fgColor="F4B183"),
    }
    writing_fills = {
        "Yes": PatternFill("solid", fgColor="FFF2CC"),
        "No": PatternFill("solid", fgColor="E2EFDA"),
    }

    headers = ["#", "Review point", "Status", "Writing only?", "What's left"]
    ws.append(headers)
    for col in range(1, 6):
        cell = ws.cell(1, col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin

    row = 2

    def add_group(title: str, items: list) -> None:
        nonlocal row
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=5)
        cell = ws.cell(row, 1, title)
        cell.font = group_font
        cell.fill = group_fill
        cell.alignment = Alignment(vertical="center")
        for c in range(1, 6):
            ws.cell(row, c).fill = group_fill
            ws.cell(row, c).border = thin
        row += 1
        for num, point, status, writing, left in items:
            ws.cell(row, 1, num).alignment = Alignment(horizontal="center", vertical="center")
            ws.cell(row, 2, point).alignment = Alignment(wrap_text=True, vertical="center")
            st = ws.cell(row, 3, status)
            st.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            if status in status_fills:
                st.fill = status_fills[status]
            wr = ws.cell(row, 4, writing)
            wr.alignment = Alignment(horizontal="center", vertical="center")
            if writing in writing_fills:
                wr.fill = writing_fills[writing]
            left_cell = ws.cell(row, 5, left)
            left_cell.alignment = Alignment(wrap_text=True, vertical="center")
            if left != "—" and status in ("Partly done", "In progress", "Not started", "Measured; no win yet", "Later"):
                left_cell.fill = PatternFill("solid", fgColor="FCE4D6")
            for c in range(1, 6):
                ws.cell(row, c).border = thin
            ws.row_dimensions[row].height = 36
            row += 1

    add_group("Reviewer 1", R1)
    add_group("Reviewer 2", R2)
    add_group("AI reviewer", AI)

    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=5)
    note = ws.cell(
        row,
        1,
        "Writing only? Yes = paper/writing only. What's left = remaining work (— if nothing left).",
    )
    note.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[row].height = 30

    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 52
    ws.column_dimensions["C"].width = 20
    ws.column_dimensions["D"].width = 12
    ws.column_dimensions["E"].width = 48
    ws.row_dimensions[1].height = 22
    ws.freeze_panes = "A2"

    wb.save(XLSX_PATH)


def set_cell_shading(cell, color_hex: str) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color_hex)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def add_table(doc: Document, headers: list, rows: list, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(8)
                run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(cell, "1F4E79")
    for r_i, row_data in enumerate(rows):
        for c_i, val in enumerate(row_data):
            cell = table.rows[r_i + 1].cells[c_i]
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", str(val))
            cell.text = text
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8)
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    return table


def write_docx() -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    doc.add_heading("AAAI → CVPR progress — details (supervisor)", level=0)
    doc.add_paragraph(
        "Detailed work, related results, BEFORE/AFTER summary tables, and remaining items. "
        "Companion Excel: AAAI_TO_CVPR_SUPERVISOR_PROGRESS_STATUS.xlsx. "
        "“Writing only?” = Yes means paper/writing only. "
        "“What’s left” = remaining work for that point (— if nothing left)."
    )

    widths = [0.3, 1.1, 1.2, 1.5, 0.85, 0.55, 1.4]
    headers = ["#", "Reviewer point", "Our work", "Related results", "Status", "Writing only?", "What's left"]

    doc.add_heading("Details — Reviewer 1", level=1)
    add_table(
        doc,
        headers,
        [
            ["1", "Own edit on wrong image not counted", "Four-way leak: stale / foreign / correct route / abstain", "Dual-Expert Keys stale ≤ 9%. Baseline VLKEB swap: foreign 82% → total leak 88%", "Done", "No", "—"],
            ["2", "Partner edit counted as error", "Partner-only fire = correct route", "Easy-VQA baseline correct route 87%, true leak 11.3%. Dual-Expert Keys leak 0.7% mainly via abstain (99%)", "Done", "No", "—"],
            ["3", "Swap-accuracy conflicted with safety", "Stopped using swap-accuracy; report how often stale vs partner answers appear", "VLKEB stale answer: baseline 8% vs Dual-Expert Keys 21% (worse). Partner answer ~0–1%", "Measured; no win yet", "No", "Improve answer-level (cut stale answers on swap); then we can claim it"],
            ["4", "Components not isolated", "Full routing-rule comparison", "VLKEB hard-neg/swap/rel: LiveEdit 93/88/100; AND alone 5/0/50; always-reject 0/0/0; Dual-Expert Keys 44/48/100", "Done", "No", "—"],
            ["5", "Binary FPR may overstate leak", "Soft-weight re-score; lead with hard-neg leak", "Soft-weight baseline VLKEB swap ~24%. Hard-neg: 91→52 (VLKEB), 96.7→9.3 (Easy-VQA)", "Done", "No", "—"],
            ["6", "Pool size untested", "Partners kept in pool; curve at 100/300/1000", "Swap leak baseline→ours: 90→19, 97→15, 100→23. Rel ≥ 99%", "Partly done", "No", "Optional: more edit orders + Easy-VQA pool-size curve"],
            ["7", "Prompt / visual collisions", "Checked whether residual leaks sit on repeated prompts", "~47% on repeated prompts (not enough to blame text gate). Hard-neg residual 94% foreign", "Partly done", "No", "Optional: hard-neg difficulty strata + small human check of pairs"],
            ["8", "One backbone / one editor", "LLaVA train on stronger GPU; other editors later", "Load check passed. Train/eval due 15 Oct. No LLaVA Dual-Expert Keys eval numbers yet. Other editors: no numbers", "In progress", "No", "Finish LLaVA Dual-Expert Keys train + eval; then other editors"],
            ["9", "Residual still high", "Marked absolute hard-neg under honest pools as open gap", "Honest-pool hard-neg Dual-Expert Keys ~84–86% (baseline 100%). Relative win only", "Measured; no win yet", "No", "Method retry to cut absolute hard-neg leak under honest pools"],
        ],
        widths,
    )

    doc.add_heading("Details — Reviewer 2", level=1)
    add_table(
        doc,
        headers,
        [
            ["1", "CARML / MSCKE missing", "Added untuned CARML-style row; MSCKE check still needed", "CARML-style VLKEB swap/hard-neg 99/100 vs Dual-Expert Keys 48/44. MSCKE: none yet", "Partly done", "No", "MSCKE public-code check + related-work write-up"],
            ["2", "Metric may be LiveEdit-only", "Same leak protocol on other editors (planned)", "No numbers yet", "Not started", "No", "Run same leak protocol on other editors"],
            ["3", "Author-built baseline only", "Original LiveEdit + AND-alone controls", "LiveEdit 88/93 @ rel 100; AND alone rel 50%; Dual-Expert Keys 48/44 @ 100%", "Done", "No", "—"],
            ["4", "No Rel/Gen/Loc gain", "Numbers ready; paper should lead with leak vs reliability", "Hard-neg @ high reliability: ours 44% / 9% vs baseline 93% / 97%. Only-own ~16%→~100%", "Partly done", "Yes", "Write paper section / plot leading with leak vs reliability"],
            ["5", "Single backbone", "LLaVA second backbone", "In progress; due 15 Oct. No LLaVA Dual-Expert Keys eval numbers yet", "In progress", "No", "Finish LLaVA Dual-Expert Keys train + eval (due 15 Oct)"],
        ],
        widths,
    )

    doc.add_heading("Details — AI reviewer", level=1)
    add_table(
        doc,
        headers,
        [
            ["1", "Conflicting metric targets", "One naming list; four-way outcomes; swap-accuracy removed", "VLKEB swap 88→20; hard-neg 91→52. Stale answers still worse (8→21)", "Partly done", "No", "Answer-level still weak; do not claim swap-answer rescue until fixed"],
            ["2", "Missing always-reject / controls", "Always-reject control + reliability / only-own / correct route", "Always-reject: rel 0%. Ours Easy-VQA abstain 99% but rel 99%. Only-own 16→100%", "Partly done", "Yes", "Paper polish: present controls clearly in tables/text"],
            ["3", "Novelty vs LiveEdit / CARML", "Ran comparison; paper side-by-side table still needed", "Beats LiveEdit (88/93), sum (68/71), CARML-style (99/100)", "Partly done", "Yes", "Write side-by-side routing-spec table in the paper"],
            ["4", "No ablation without adversarial / vision / text", "Ran those rows", "Vision≈LiveEdit; text-only swap 99%; AND alone rel 50%/61%", "Done", "No", "—"],
            ["5", "Lifelong scaling untested", "Pool curve with partners in pool (VLKEB, one seed)", "Swap ours 19/15/23 vs baseline 90/97/100. Hard-neg ours ~84–86%", "Done", "No", "—"],
            ["6", "Prototype score underspecified", "Needs clear write-up in the paper", "—", "Not started", "Yes", "Write concrete prototype-score definition in method section"],
            ["7", "Hard-neg strata + human check", "Put off after residual check", "—", "Later", "No", "Optional later if time remains"],
            ["8", "VLKEB vs Easy-VQA residual", "Short residual check", "VLKEB repeated-prompt rate 40%, residual 20/52; Easy-VQA 2.7%, residual 0.7/9.3", "Done", "No", "—"],
            ["9", "Multi-seed shared checkpoints", "Five eval seeds planned", "Not run yet", "Not started", "No", "Run five evaluation seeds (and training seeds if time)"],
            ["10", "Hard-neg missing from main tables", "Hard-neg is now the main leak number", "Honest 91→52; honest-pool curve ~84% (still weak in absolute terms)", "Done", "No", "—"],
            ["11", "Writing nits", "Naming rules locked", "Paper draft cleanup still due", "Partly done", "Yes", "Clean draft: spelling, terms, figure seed labels, report hyperparameters"],
        ],
        widths,
    )

    doc.add_heading("Summary results (BEFORE vs AFTER)", level=1)
    doc.add_paragraph(
        "Lower is better for leak. Higher is better for correct route and only-own. All figures in %."
    )

    doc.add_heading("Leakage metric", level=2)
    add_table(
        doc,
        ["When", "Dataset", "Method", "Swap leak ↓", "Correct route ↑", "Hard-neg leak ↓", "Only-own ↑"],
        [
            ["BEFORE (AAAI)", "Easy-VQA-style", "Dual-Expert Keys", "Claimed ~96–98% “false activation” fixed (mixed correct route into error)", "—", "Not in VLKEB main table", "—"],
            ["AFTER", "VLKEB", "Baseline", "88.0", "8.0", "91.0", "16.0"],
            ["AFTER", "VLKEB", "Dual-Expert Keys", "20.0", "34.0", "52.0", "100.0"],
            ["AFTER", "Easy-VQA", "Baseline", "11.3", "87.0", "96.7", "17.3"],
            ["AFTER", "Easy-VQA", "Dual-Expert Keys", "0.7", "0.3", "9.3", "99.0"],
        ],
    )

    doc.add_heading("Routing controls (VLKEB) — AFTER only (BEFORE: not run)", level=2)
    add_table(
        doc,
        ["When", "Rule", "Swap leak ↓", "Hard-neg leak ↓", "Rel hit ↑"],
        [
            ["AFTER", "Original LiveEdit", "88", "93", "100"],
            ["AFTER", "AND alone", "0", "5", "50"],
            ["AFTER", "Always reject", "0", "0", "0"],
            ["AFTER", "Untuned CARML-style", "99", "100", "100"],
            ["AFTER", "Dual-Expert Keys", "48", "44", "100"],
        ],
    )

    doc.add_heading("Lifelong pool (VLKEB, partners in pool) — AFTER only (BEFORE: ~100 edits only)", level=2)
    add_table(
        doc,
        ["When", "Pool size", "Baseline swap / hard-neg ↓", "Dual-Expert Keys swap / hard-neg ↓", "Correct route ↑", "Rel ↑"],
        [
            ["AFTER", "100", "90 / 100", "19 / 84", "68", "100"],
            ["AFTER", "300", "97 / 100", "15 / 86", "70", "99.7"],
            ["AFTER", "1000", "100 / 100", "23 / 84", "62", "99.2"],
        ],
    )

    doc.add_heading("Answer-level (AFTER; BEFORE used conflicting swap-accuracy)", level=2)
    add_table(
        doc,
        ["When", "Dataset", "Method", "Stale answer ↓", "Partner answer ↑"],
        [
            ["AFTER", "VLKEB", "Baseline", "8.0", "0.0"],
            ["AFTER", "VLKEB", "Dual-Expert Keys", "21.0 (worse)", "0.0"],
            ["AFTER", "Easy-VQA", "Baseline", "8.3", "0.7"],
            ["AFTER", "Easy-VQA", "Dual-Expert Keys", "9.3", "1.3"],
        ],
    )

    doc.add_heading("Still pending", level=1)
    add_table(
        doc,
        ["Item", "Timing"],
        [
            ["LLaVA Dual-Expert Keys train + eval", "15 Oct"],
            ["Other editors under same leak protocol", "After backbone"],
            ["Absolute hard-neg method retry", "After backbone"],
            ["Multi-seed runs", "After configs locked"],
            ["MSCKE check + paper writing", "Late Oct / 27 Oct"],
            ["Experiment freeze", "5 Nov"],
        ],
        [4.5, 2.0],
    )

    doc.save(DOCX_PATH)


if __name__ == "__main__":
    write_xlsx()
    write_docx()
    print("wrote", XLSX_PATH)
    print("wrote", DOCX_PATH)
