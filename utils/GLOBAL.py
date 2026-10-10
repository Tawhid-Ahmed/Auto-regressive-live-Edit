import os

# Project root = this repo (folder containing utils/). Override with LIVEEDIT_ROOT if needed.
ROOT_PATH = os.environ.get(
    "LIVEEDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

# Model paths relative to ROOT_PATH. BLIP2/LLaVA often live in ../models/ next to this repo
# (e.g. .../LiveEdit/models/llava-1.5-7b-hf while ROOT_PATH is .../LiveEdit/LiveEdit).
_model_rel_paths = {
    'blip2-opt-2.7b': os.path.join('..', 'models', 'blip2-opt-2.7b'),
    'llava-v1.5-7b': os.path.join('..', 'models', 'llava-1.5-7b-hf'),
    'minigpt-4-vicuna-7b': 'models/minigpt-4-vicuna-7b',
}
model_path_map = {
    k: os.path.normpath(os.path.join(ROOT_PATH, v)) for k, v in _model_rel_paths.items()
}
