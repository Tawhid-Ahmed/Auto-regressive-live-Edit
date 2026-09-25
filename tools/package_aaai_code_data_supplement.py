"""Build an anonymous AAAI-27 Code and Data Supplement folder.

Usage (from repo root):
  python tools/package_aaai_code_data_supplement.py

Output:
  AuthorKit27/Dual-Expert-Keys/

Then ZIP that folder (must stay under OpenReview's 50MB limit) and upload
as the Code and Data Supplement. Do not include GitHub / Hugging Face links.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "AuthorKit27" / "Dual-Expert-Keys"
README_TEMPLATE = ROOT / "AuthorKit27" / "code_supplement_README.md"

# Skip caches and identity-heavy / oversized material.
SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".cursor",
    ".vscode",
    "records",
    "eval_results",
    "AuthorKit27",
    "doc",
    "docs",
    "figures",
}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIR_NAMES:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    name = path.name.lower()
    if "thesis" in name or "supervisor" in name:
        return True
    return False


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[skip missing] {src}")
        return
    for path in src.rglob("*"):
        if should_skip(path):
            continue
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            copy_and_sanitize_file(path, target)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\n"), encoding="utf-8")


def sanitize_text(text: str) -> str:
    """Strip personal absolute paths that would break double-blind review."""
    # Windows absolute roots used in this workspace
    replacements = [
        (r"E:\MSC\MSc thesis project\LiveEdit\LiveEdit", "."),
        (r"E:/MSC/MSc thesis project/LiveEdit/LiveEdit", "."),
        (r"E:\\MSC\\MSc thesis project\\LiveEdit\\LiveEdit", "."),
        (r"E:\MSC\MSc thesis project\LiveEdit", "."),
        (r"E:/MSC/MSc thesis project/LiveEdit", "."),
        (r"E:\\MSC\\MSc thesis project\\LiveEdit", "."),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    # Neutralize internal thesis/meeting workflow naming
    text = text.replace(
        "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
        "data/VLKEB/grounding/grounding_eval_short.json",
    )
    text = text.replace(
        "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json",
        "data/VLKEB/grounding/grounding_eval_short.json",
    )
    text = text.replace("doc/results/meeting_validation/step1/", "eval_results/grounding/")
    text = text.replace("eval_results/meeting_validation/step", "eval_results/grounding_step")
    text = text.replace("records/meeting_validation/step6", "records/liveedit")
    text = text.replace("../vlkeb_long_ollama/out/eval_longform_ollama.json", "data/VLKEB/eval.json")
    text = text.replace("meeting_validation", "grounding")
    return text


def copy_and_sanitize_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".md", ".txt"}:
        try:
            text = src.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(src, dst)
            return
        dst.write_text(sanitize_text(text), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # Core code trees
    for name in ("editor", "dataset", "evaluation", "utils", "configs"):
        copy_tree(ROOT / name, OUT / name)

    # Entry scripts
    for name in (
        "train_vllm_editor.py",
        "test_vllm_edit.py",
        "test_grounding_eval.py",
    ):
        src = ROOT / name
        if src.exists():
            copy_and_sanitize_file(src, OUT / name)

    # Essential tools only (not the full tools/ tree — too large / thesis-specific)
    tools_out = OUT / "tools"
    tools_out.mkdir(parents=True, exist_ok=True)
    for rel in (
        "build_grounding_eval_json.py",
        "analyze_grounding_results.py",
        "log_grounding_diagnostics.py",
        "log_routing_diagnostics.py",
    ):
        src = ROOT / "tools" / rel
        if src.exists():
            copy_and_sanitize_file(src, tools_out / rel)

    # Probe-augmented JSON (representative; images must be obtained separately)
    data_copies = [
        (
            ROOT / "data" / "VLKEB" / "grounding" / "grounding_train_short.json",
            OUT / "data" / "VLKEB" / "grounding" / "grounding_train_short.json",
        ),
        (
            ROOT
            / "data"
            / "VLKEB"
            / "meeting_validation"
            / "step1"
            / "grounding"
            / "grounding_eval_short.json",
            OUT / "data" / "VLKEB" / "grounding" / "grounding_eval_short.json",
        ),
        (
            ROOT / "data" / "easy-edit-mm" / "grounding" / "grounding_train_short.json",
            OUT / "data" / "easy-edit-mm" / "grounding" / "grounding_train_short.json",
        ),
        (
            ROOT / "data" / "easy-edit-mm" / "grounding" / "grounding_eval_short.json",
            OUT / "data" / "easy-edit-mm" / "grounding" / "grounding_eval_short.json",
        ),
    ]
    for src, dst in data_copies:
        if src.exists():
            copy_and_sanitize_file(src, dst)
            print(f"[data] {src.name} -> {dst.relative_to(OUT)}")
        else:
            print(f"[warn] missing data: {src}")

    # Prefer relative defaults in packaged eval entrypoint
    eval_entry = OUT / "test_grounding_eval.py"
    if eval_entry.exists():
        text = eval_entry.read_text(encoding="utf-8")
        text = text.replace(
            "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
            "data/VLKEB/grounding/grounding_eval_short.json",
        )
        eval_entry.write_text(text, encoding="utf-8")

    # Relative model paths for SERAC configs (no personal drive letters)
    for cfg in (OUT / "configs").rglob("*.yaml"):
        text = cfg.read_text(encoding="utf-8")
        new = (
            text.replace("./models/opt-125m", "models/opt-125m")
            .replace("./models/bert-base-cased", "models/bert-base-cased")
            .replace("E:/MSC/MSc thesis project/LiveEdit/models/opt-125m", "models/opt-125m")
            .replace("E:/MSC/MSc thesis project/LiveEdit/models/bert-base-cased", "models/bert-base-cased")
        )
        # After sanitize_text, paths may become "./models/..."
        if "models/opt-125m" in new or "models/bert-base-cased" in new:
            new = new.replace('"./models/opt-125m"', '"models/opt-125m"')
            new = new.replace('"./models/bert-base-cased"', '"models/bert-base-cased"')
        if new != text:
            cfg.write_text(new, encoding="utf-8")
            print(f"[sanitize cfg] {cfg.relative_to(OUT)}")

    # Anonymous GLOBAL.py (no personal absolute path)
    write_text(
        OUT / "utils" / "GLOBAL.py",
        '''
import os

# Set this to the absolute path of this package root before running.
ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Model paths relative to ROOT_PATH. Download BLIP-2 OPT-2.7B weights separately
# and place them under models/blip2-opt-2.7b (or update the path below).
_model_rel_paths = {
    "blip2-opt-2.7b": os.path.join("models", "blip2-opt-2.7b"),
    "llava-v1.5-7b": os.path.join("models", "llava-1.5-7b-hf"),
    "minigpt-4-vicuna-7b": os.path.join("models", "minigpt-4-vicuna-7b"),
}
model_path_map = {
    k: os.path.normpath(os.path.join(ROOT_PATH, v)) for k, v in _model_rel_paths.items()
}
''',
    )

    write_text(
        OUT / "requirements.txt",
        '''
torch>=2.0
transformers>=4.36
accelerate
datasets
Pillow
numpy
pyyaml
tqdm
timm
einops
sentencepiece
protobuf
scikit-learn
''',
    )

    write_text(
        OUT / "data" / "README.md",
        '''
# Data notes (anonymous)

This package includes **probe-augmented JSON metadata only**.

Images are **not** redistributed here (size + license). To run experiments:

1. Obtain the original **VLKEB** dataset and images from the VLKEB authors / release.
2. Place VLKEB images so that paths in the JSON resolve under:
   `data/VLKEB/VLKEB_images/mmkb_images/`
3. For EVQA stress, obtain EasyEdit multimodal / EVQA + COCO images and place them
   according to the EasyEdit layout under `data/easy-edit-mm/`.

Included JSON files:
- `data/VLKEB/grounding/grounding_train_short.json`
- `data/VLKEB/grounding/grounding_eval_short.json`
- `data/easy-edit-mm/grounding/grounding_train_short.json`
- `data/easy-edit-mm/grounding/grounding_eval_short.json`

Rebuild probes from original VLKEB/EVQA splits with:
`python tools/build_grounding_eval_json.py ...`
''',
    )

    if README_TEMPLATE.exists():
        shutil.copy2(README_TEMPLATE, OUT / "README.md")
    elif (OUT / "README.md").exists():
        pass
    else:
        write_text(OUT / "README.md", "# Code and Data Supplement\n\nSee paper supplement instructions.\n")

    # Size report
    total = 0
    n_files = 0
    for p in OUT.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            n_files += 1
    print(f"\nBuilt: {OUT}")
    print(f"Files: {n_files}")
    print(f"Size:  {total / (1024 * 1024):.2f} MB (ZIP will be smaller if compressed)")
    if total > 50 * 1024 * 1024:
        print("WARNING: uncompressed size already exceeds 50MB; trim data before zipping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
