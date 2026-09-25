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
