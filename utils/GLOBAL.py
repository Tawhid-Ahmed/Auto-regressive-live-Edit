import os

# Set this to your project root (the LiveEdit folder). Model paths are relative to it.
ROOT_PATH = r'E:\MSC\MSc thesis project\LiveEdit\LiveEdit'

# Model paths relative to ROOT_PATH (do not use absolute paths here)
_model_rel_paths = {
    'blip2-opt-2.7b': 'models/blip2-opt-2.7b',
    'llava-v1.5-7b': 'models/llava-v1.5-7b-hf',
    'minigpt-4-vicuna-7b': 'models/minigpt-4-vicuna-7b',
}
model_path_map = {k: os.path.join(ROOT_PATH, v) for k, v in _model_rel_paths.items()}
