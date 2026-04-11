#%%
from utils import get_full_model_name, load_vllm_editor
from evaluation.vllm_editor_eval import VLLMEditorEvaluation
from utils.GLOBAL import ROOT_PATH
from datetime import datetime
import os, argparse, time, json

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--editor_name', type=str, help='Editor name: LiveEdit, FT_VL...', required=True)
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: llava...', required=True)
    parser.add_argument('-sen', '--sequential_edit_n', type=int, help='Edit number.', required=True)
    parser.add_argument('-enp', '--eval_name_postfix', type=str, default = '', help='Optional tag in the run folder name (each run also gets a timestamp).')
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    parser.add_argument('-ckpt', '--editor_ckpt_path', type=str, default = None, help='For Editors that needs training.')
    parser.add_argument('-dn', '--data_name', type=str, required = True, help = 'Evaluating dataset, including EVQA, EIC.')
    parser.add_argument('-dsn', '--data_sample_n', type=int, default = None, help = 'Sample number for evaluation.')
    parser.add_argument('-seed', '--eval_seed', type=int, default=None, help = 'Fixed seed for eval data shuffle (enables reproducible split for comparison).')
    # AR-LiveEdit (no behavior change when ar_mode=False)
    parser.add_argument('--ar_mode', action='store_true', help='Enable autoregressive chunk-wise edit mode.')
    parser.add_argument('--chunk_size', type=int, default=16, help='Target chunk size in tokens for AR mode.')
    parser.add_argument('--max_chunks', type=int, default=None, help='Optional max chunks per target in AR mode (default: unlimited).')
    args = parser.parse_args()
    # Treat common sentinels as "no checkpoint"
    if isinstance(args.editor_ckpt_path, str) and args.editor_ckpt_path.strip().lower() in {"none", "null", ""}:
        args.editor_ckpt_path = None
    return args
 
class cfg:
    editor_name = 'liveedit'
    edit_model_name = 'blip2'
    sequential_edit_n = 100
    eval_name_postfix = 'test'
    device = 0
    editor_ckpt_path = 'records/...'
    data_name = 'EVQA' # 'EVQA', 'EIC'
    data_sample_n = 100 # 231, 530, 1300, 3000
    # AR-LiveEdit
    ar_mode = False
    chunk_size = 16
    max_chunks = None

if __name__ == '__main__':
    cfg = get_attr()
    cfg.editor_name = cfg.editor_name.lower()
    cfg.edit_model_name = get_full_model_name(cfg.edit_model_name)
    # Unique folder per run: DATA[-tag][-ar]-YYYY.MM.DD-HH.MM.SS (matches training record style)
    _parts = [cfg.data_name.upper()]
    if cfg.ar_mode:
        _parts.append('ar')
    if cfg.eval_name_postfix:
        _parts.append(cfg.eval_name_postfix)
    _parts.append(datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))
    cfg.evaluation_name = '-'.join(_parts)
    print(cfg)
    editor = load_vllm_editor(cfg.editor_name, cfg.edit_model_name, cfg.device, None, cfg.editor_ckpt_path, False)
    # AR-LiveEdit: set editor flags so edit_one_piece uses chunk-wise path when --ar_mode
    if hasattr(editor, 'ar_mode'):
        editor.ar_mode = getattr(cfg, 'ar_mode', False)
        editor.chunk_size = getattr(cfg, 'chunk_size', 16)
        editor.max_chunks = getattr(cfg, 'max_chunks', None)
    # load data
    if cfg.data_name == 'EVQA':
        from dataset.vllm import EVQA
        data_path = os.path.join(ROOT_PATH, 'data/easy-edit-mm/vqa/vqa_eval.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/easy-edit-mm')
        eval_data = EVQA(data_path, img_root_dir, cfg.data_sample_n)
    elif cfg.data_name == 'EIC':
        from dataset.vllm import EIC
        data_path = os.path.join(ROOT_PATH, 'data/easy-edit-mm/caption/caption_eval_edit.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/easy-edit-mm')
        eval_data = EIC(data_path, img_root_dir, cfg.data_sample_n)
    elif cfg.data_name == 'VLKEB':
        from dataset.vllm import VLKEB
        data_path = os.path.join(ROOT_PATH, 'data/VLKEB/eval.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/VLKEB/VLKEB_images/mmkb_images')
        eval_data = VLKEB(data_path, img_root_dir, cfg.data_sample_n)
    # evaluate (use fixed seed when provided for reproducible baseline vs AR comparison)
    use_random = getattr(cfg, 'eval_seed', None) is not None
    eval_seed = getattr(cfg, 'eval_seed', None)
    ev = VLLMEditorEvaluation(editor, eval_data, cfg.evaluation_name, 'eval_results', ar_chunk_size=getattr(cfg, 'chunk_size', 16))
    t0 = time.time()
    ev.evaluate_sequential_edit(cfg.sequential_edit_n, use_random, eval_seed)
    elapsed = time.time() - t0
    mean_results_fname = ('seed_%s_mean_results.json' % eval_seed) if use_random else 'mean_results.json'
    mean_results_path = os.path.join(ev.result_dir, 'sequential_edit_%s' % cfg.sequential_edit_n, mean_results_fname)
    if os.path.exists(mean_results_path):
        with open(mean_results_path, 'r') as f:
            data = json.load(f)
        data['runtime_seconds'] = round(elapsed, 2)
        data['runtime_minutes'] = round(elapsed / 60, 2)
        with open(mean_results_path, 'w') as f:
            json.dump(data, f, indent=4)
    print('[Timer] Total elapsed: %.2f s (%.2f min)' % (elapsed, elapsed / 60))

