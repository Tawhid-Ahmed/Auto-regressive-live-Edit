"""Generate supervisor thesis report DOCX from doc/THESIS_SUPERVISOR_REPORT.md."""
from __future__ import annotations

from pathlib import Path

from build_thesis_presentation_v2_docx import build

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    build(
        src=ROOT / "doc" / "THESIS_SUPERVISOR_REPORT.md",
        out=ROOT / "doc" / "THESIS_SUPERVISOR_REPORT.docx",
        img_dir=ROOT / "doc" / "_supervisor_report_diagrams",
    )
