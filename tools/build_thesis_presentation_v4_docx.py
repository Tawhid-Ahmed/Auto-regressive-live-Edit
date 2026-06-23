"""Generate thesis presentation DOCX (V4) from doc/THESIS_PRESENTATION_V4.md."""
from __future__ import annotations

from pathlib import Path

from build_thesis_presentation_v2_docx import build

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    build(
        src=ROOT / "doc" / "THESIS_PRESENTATION_V4.md",
        out=ROOT / "doc" / "THESIS_PRESENTATION_V4.docx",
        img_dir=ROOT / "doc" / "_presentation_v4_diagrams",
    )
