"""Generate a formal, shareable thesis progress report (DOCX)."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "doc"
SAMPLES = DOC / "figures" / "supervisor_samples"
FIG = DOC / "results" / "meeting_validation" / "step6" / "figures"
OUT = DOC / "THESIS_PROGRESS_REPORT_FORMAL.docx"


def set_run_font(run, size=11, bold=False, italic=False, color=None):
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run_font(run, size=16 if level == 1 else 13 if level == 2 else 12, bold=True)
    return h


def add_para(doc, text, *, bold=False, italic=False, size=11, space_after=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(item)
        set_run_font(run, size=11)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        set_run_font(cell.paragraphs[0].add_run(h), size=10, bold=True)
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = table.rows[r_i + 1].cells[c_i]
            cell.text = ""
            set_run_font(cell.paragraphs[0].add_run(str(val)), size=10)
    doc.add_paragraph()


def add_image(doc, path: Path, width=4.5, caption=None):
    if not path.is_file():
        add_para(doc, f"[Figure unavailable: {path.name}]", italic=True, size=10)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(10)
        set_run_font(
            cap.add_run(caption),
            size=9,
            italic=True,
            color=RGBColor(0x55, 0x55, 0x55),
        )


def main():
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(0.9)
        s.bottom_margin = Inches(0.9)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(title.add_run("Progress Report"), size=20, bold=True)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(
        sub.add_run(
            "Dual-Expert Keys for Reducing False-Cross Retrieval\n"
            "in Lifelong Vision–Language Model Editing"
        ),
        size=13,
        bold=True,
    )

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(
        meta.add_run(
            "MSc Thesis Project  ·  July 2026\n"
            "Base system: LiveEdit (Chen et al., CVPR 2025) on BLIP2-OPT-2.7B\n"
            "Proposed method: Dual-Expert Keys (DEK), with AVBT as training support"
        ),
        size=10,
        italic=True,
        color=RGBColor(0x44, 0x44, 0x44),
    )

    # Abstract-like intro
    add_heading(doc, "1. Overview", 1)
    add_para(
        doc,
        "Lifelong vision–language model (VLM) editors maintain a growing pool of edit "
        "experts and retrieve them at inference time. Prior work, notably LiveEdit, "
        "demonstrates strong reliability, generality, and locality under standard "
        "evaluation. This project studies a failure mode that those metrics largely "
        "omit: under adversarial image-swap and hard-negative probes, the editor often "
        "retrieves a foreign expert from the lifelong pool. We term this rate "
        "false-cross (FC).",
    )
    add_para(
        doc,
        "The proposed method, Dual-Expert Keys (DEK), retrieves an expert only when "
        "vision and text routing signals agree (vision ∧ text). On short VLKEB grounding "
        "evaluation, DEK reduces FC from 96% to 46% while preserving reliability near "
        "94%. On fair long-form evaluation with matched training, FC falls from 88% to "
        "38% with reliability near 73%. Ablations show that adversarial visual binding "
        "training (AVBT) alone does not reduce FC; the dual-gate is the effective "
        "component.",
    )

    # Contribution
    add_heading(doc, "2. Contribution", 1)
    add_para(doc, "The contribution of this work is threefold:")
    add_bullets(
        doc,
        [
            "Problem: identify false-cross expert retrieval as a dominant form of visual grounding leakage in lifelong MoE VLM editors, under image-swap / hard-negative probes that standard Rel/Gen/Loc tables do not emphasise.",
            "Method: Dual-Expert Keys — store separate vision and text keys per expert and admit retrieval only under vision ∧ text agreement.",
            "Evidence: ablations, five-seed short evaluation with paired significance, external editors (FT / MEND / SERAC), and fair long-form comparison against a LiveEdit baseline.",
        ],
    )
    add_para(
        doc,
        "This work extends the LiveEdit architecture; it does not replace LiveEdit as a "
        "from-scratch lifelong VLM editor. Related text-only long-form editing work "
        "(e.g. AnyEdit) is orthogonal and is cited for context only.",
        italic=True,
    )

    # Setup
    add_heading(doc, "3. Experimental setup", 1)
    add_table(
        doc,
        ["Item", "Setting"],
        [
            ["Backbone", "Frozen BLIP2-OPT-2.7B"],
            ["Editor", "LiveEdit MoE lifelong pool"],
            ["Training / eval data", "VLKEB grounding short and long"],
            ["Default protocol", "dsn=100, sen=10"],
            ["Primary metrics", "Reliability (Rel); False-cross (FC) ↓; Image-swap token match (leakage)"],
            ["Proposed configuration", "--patch_grounding --dek_dual_gate --log_routing"],
        ],
    )

    add_heading(doc, "3.1 Metric definitions", 2)
    add_table(
        doc,
        ["Metric", "Definition", "Preferred direction"],
        [
            [
                "Reliability (Rel)",
                "Token accuracy of the edit target on the correct image",
                "Higher",
            ],
            [
                "Image-swap",
                "Token match to the edit target when the image is replaced by a wrong-entity partner",
                "Lower (less leakage)",
            ],
            [
                "Hard-negative",
                "Same scoring on a visually confusable partner image",
                "Lower (less leakage)",
            ],
            [
                "False-cross (FC)",
                "Rate at which a non-own expert is retrieved under swap / hard-neg routing logs",
                "Lower",
            ],
        ],
    )
    add_para(
        doc,
        "Note on image-swap: the probe scores how much of the edited answer still appears "
        "on a wrong image. Therefore a higher percentage indicates stronger token-level "
        "leakage, not stronger grounding.",
        italic=True,
    )

    # Sample walkthrough
    add_heading(doc, "4. Method illustrated on a data sample", 1)
    add_para(
        doc,
        "The following example is taken from the short grounding evaluation set "
        "(index 0).",
    )
    add_table(
        doc,
        ["Field", "Value"],
        [
            ["Prompt", "Which Illinois city is shown in the picture?"],
            ["Pre-edit belief", "Laredo, Texas"],
            ["Edit target", "Kankakee"],
        ],
    )
    add_image(
        doc,
        SAMPLES / "sampleA_montage.jpg",
        width=6.0,
        caption="Figure 1. Sample probes: edit image, image-swap partner, hard-negative, and modality-consistency view.",
    )

    add_heading(doc, "4.1 Edit write", 2)
    add_para(
        doc,
        "At edit time, LiveEdit creates an MoE expert e_Kankakee and stores retrieval "
        "keys derived from the correct image and prompt. The correct image used for the "
        "vision key is shown below.",
    )
    add_image(
        doc,
        SAMPLES / "sampleA_edit_kankakee.jpg",
        width=3.2,
        caption="Figure 2. Correct edit image (Reliability: expect “Kankakee”).",
    )

    add_heading(doc, "4.2 AVBT training (support)", 2)
    add_para(
        doc,
        "AVBT adds a training-time binding objective: strengthen the own expert on the "
        "correct image, and penalise binding on swap / hard-negative images.",
    )
    add_image(doc, SAMPLES / "sampleA_edit_kankakee.jpg", width=2.8, caption="Figure 3a. Positive pair (correct image).")
    add_image(doc, SAMPLES / "sampleA_swap.jpg", width=2.8, caption="Figure 3b. Negative pair (image-swap).")
    add_image(doc, SAMPLES / "sampleA_hardneg.jpg", width=2.8, caption="Figure 3c. Negative pair (hard-negative).")
    add_para(
        doc,
        "Empirically, AVBT alone does not reduce false-cross (FC remains ~96%). It is "
        "retained as a training component; Dual-Expert Keys is the mechanism that "
        "improves FC.",
        italic=True,
    )

    add_heading(doc, "4.3 Dual-Expert Keys at retrieval", 2)
    add_para(
        doc,
        "Each expert stores separate vision and text keys. At inference, an expert is "
        "kept only if vision_ok ∧ text_ok.",
    )
    add_para(
        doc,
        "On the correct image, both gates typically pass and e_Kankakee is applied "
        "(Reliability). On the swap image below, the dual-gate is intended to block "
        "e_Kankakee even if the prompt resembles the edit, thereby reducing false-cross.",
    )
    add_image(
        doc,
        SAMPLES / "sampleA_swap.jpg",
        width=3.2,
        caption="Figure 4. Image-swap query: wrong entity with an edit-like prompt.",
    )
    arch = DOC / "THESIS_architecture_AVBT_DEK.png"
    if arch.is_file():
        add_image(
            doc,
            arch,
            width=5.8,
            caption="Figure 5. Method architecture overview.",
        )

    # Results
    add_heading(doc, "5. Results", 1)
    add_para(
        doc,
        "Unless noted otherwise, results use dsn=100, sen=10, and the short or long "
        "grounding evaluation JSON under the project meeting-validation protocol.",
        size=10,
        italic=True,
    )

    add_heading(doc, "5.1 Component ablation (short, seed 42)", 2)
    add_table(
        doc,
        ["Method", "Rel ↑", "Image-swap (leakage)", "FC ↓", "Interpretation"],
        [
            ["LiveEdit baseline (L0)", "93.5%", "40.7%", "96%", "High Rel; high FC"],
            ["AVBT only (S2)", "93.8%", "47.8%", "96%", "FC unchanged; swap leakage ↑"],
            ["AVBT + Dual-Expert Keys (DEK)", "93.7%", "47.7%", "46%", "FC cut; Rel held"],
            ["+ False-cross mining (FX)", "93.7%", "47.2%", "43%", "No clear swap gain"],
            ["+ Pool buffer (PB)", "93.7%", "48.0%", "55%", "FC regresses"],
        ],
    )
    add_para(
        doc,
        "Conclusion from ablation: Dual-Expert Keys is necessary for the FC reduction. "
        "Additional mining variants do not establish a stronger primary method.",
    )
    if (FIG / "aaai_ablation_short.png").is_file():
        add_image(doc, FIG / "aaai_ablation_short.png", width=5.6, caption="Figure 6. Short ablation.")

    add_heading(doc, "5.2 Multi-seed evaluation (short)", 2)
    add_para(
        doc,
        "Seeds {42, 123, 456, 789, 1024}; paired Student t-tests over evaluation seeds.",
    )
    add_table(
        doc,
        ["Metric", "LiveEdit baseline", "Dual-Expert Keys", "Change"],
        [
            ["Reliability", "93.5%", "93.7%", "Held"],
            ["False-cross", "96%", "46%", "−50 pp (stable across seeds)"],
            ["Image-swap (leakage)", "41.2%", "48.6%", "+7.4 pp (see metric note above)"],
        ],
    )
    if (FIG / "aaai_multiseed_dek.png").is_file():
        add_image(doc, FIG / "aaai_multiseed_dek.png", width=5.6, caption="Figure 7. Multi-seed comparison.")

    add_heading(doc, "5.3 Comparison with other editors (short)", 2)
    add_table(
        doc,
        ["Method", "Rel", "Image-swap", "FC"],
        [
            ["LiveEdit baseline (L0)", "93.5%", "40.7%", "96%"],
            ["Dual-Expert Keys (DEK)", "93.7%", "47.7%", "46%"],
            ["Fine-tuning (FT)", "73.0%", "73.1%", "n/a"],
            ["MEND-VL", "71.3%", "69.4%", "n/a"],
            ["SERAC-VL", "38.6%", "38.6%", "n/a"],
        ],
    )
    add_para(
        doc,
        "FT and MEND exhibit lower reliability with image-swap close to Rel (weak / "
        "soft-edit operating point). False-cross is defined only for LiveEdit-style "
        "retrieval pools.",
    )
    if (FIG / "aaai_competitors_short.png").is_file():
        add_image(doc, FIG / "aaai_competitors_short.png", width=5.6, caption="Figure 8. Competitor comparison.")

    add_heading(doc, "5.4 Fair long-form evaluation", 2)
    add_para(
        doc,
        "Both models are trained on long-form data and evaluated on long-form grounding "
        "(matched training). Short-to-long transfer without retraining is not used as "
        "the headline long-form result.",
    )
    add_table(
        doc,
        ["Method", "Rel", "Image-swap", "FC"],
        [
            ["LiveEdit baseline (long)", "74.0%", "42.4%", "88%"],
            ["Dual-Expert Keys (long)", "72.8%", "42.7%", "38%"],
        ],
    )
    add_para(
        doc,
        "The long-form pattern matches the short-form story: substantial FC reduction "
        "with reliability largely preserved; image-swap remains near the mid-40% range.",
    )
    if (FIG / "aaai_long_fair.png").is_file():
        add_image(doc, FIG / "aaai_long_fair.png", width=5.2, caption="Figure 9. Fair long-form results.")

    add_heading(doc, "5.5 Negative findings", 2)
    add_bullets(
        doc,
        [
            "Hyperparameter sweeps on the AVBT loss weight did not yield a breakthrough on token-level swap leakage.",
            "False-cross mining and pool-buffer extensions did not improve the primary operating point beyond Dual-Expert Keys.",
            "A stretch target of ≥55% long-form image-swap (under earlier alternative framing) was not achieved; the defended claim is FC reduction with Rel preserved.",
        ],
    )
    if (FIG / "aaai_results_panel.png").is_file():
        add_image(doc, FIG / "aaai_results_panel.png", width=5.8, caption="Figure 10. Summary results panel.")

    # Status / next
    add_heading(doc, "6. Current status and planned work", 1)
    add_para(doc, "Completed:")
    add_bullets(
        doc,
        [
            "Grounding diagnosis (routing bottleneck and false-cross characterisation).",
            "Implementation of Dual-Expert Keys and supporting training flags.",
            "Short ablations, five-seed evaluation, external editor comparison, and fair long-form evaluation.",
            "Documentation of architecture, sample walkthrough, and result figures.",
        ],
    )
    add_para(doc, "Planned:")
    add_bullets(
        doc,
        [
            "Side-by-side LiveEdit-native generality and locality metrics (T-Gen / M-Gen / T-Loc / M-Loc) for baseline vs Dual-Expert Keys.",
            "Human audit of a sampled subset of image-swap / hard-negative pairs.",
            "Thesis write-up centred on false-cross and Dual-Expert Keys.",
            "Optional: second backbone and larger sequential-edit stress settings, subject to time.",
        ],
    )

    add_heading(doc, "7. Summary", 1)
    add_para(
        doc,
        "Lifelong VLM editors can maintain high reliability while still retrieving wrong "
        "experts under adversarial images. Dual-Expert Keys address this false-cross "
        "failure by requiring vision–text agreement at retrieval. Across short multi-seed "
        "and fair long-form experiments, false-cross is reduced substantially and "
        "reliability is preserved. The work is positioned as an extension of LiveEdit "
        "focused on visual grounding leakage, with Dual-Expert Keys as the primary "
        "methodological contribution.",
    )

    doc.save(OUT)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
