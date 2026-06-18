#%%
"""
Task 8: First LiveEdit vs AR-LiveEdit controlled comparison on VLKEB.
Runs baseline (ar_mode=False) and AR (ar_mode=True) eval on the same split/seed,
then compares metrics and validates long-form gain and locality bounded.
"""
from utils import get_full_model_name, load_vllm_editor
from evaluation.vllm_editor_eval import VLLMEditorEvaluation
from utils.GLOBAL import ROOT_PATH
import os
import sys
import json
import argparse
import time


def get_attr():
    parser = argparse.ArgumentParser(description='LiveEdit vs AR-LiveEdit controlled comparison on VLKEB.')
    parser.add_argument('-dvc', '--device', type=str, default='cuda:0', help='CUDA device.')
    parser.add_argument('-ckpt', '--editor_ckpt_path', type=str, default=None, help='Checkpoint path (used for both if --ar_ckpt not set).')
    parser.add_argument('--ar_ckpt', type=str, default=None, help='AR checkpoint path (default: same as --editor_ckpt_path).')
    parser.add_argument('-dsn', '--data_sample_n', type=int, default=200, help='VLKEB eval sample count (use 3174 for full long-form eval).')
    parser.add_argument(
        '-dpath',
        '--vlkeb_eval_json',
        type=str,
        default=None,
        help='VLKEB eval JSON path (default: data/VLKEB/eval.json). Use long-form eval JSON for thesis runs.',
    )
    parser.add_argument('-sen', '--sequential_edit_n', type=int, default=50, help='Edits per sequential batch.')
    parser.add_argument('-seed', '--eval_seed', type=int, default=42, help='Fixed seed for same split (required for comparison).')
    parser.add_argument('--chunk_size', type=int, default=16, help='Chunk size for AR mode.')
    parser.add_argument('--max_chunks', type=int, default=None, help='Max chunks for AR (default: unlimited).')
    parser.add_argument('--run_baseline_only', action='store_true', help='Only run baseline eval.')
    parser.add_argument('--run_ar_only', action='store_true', help='Only run AR eval.')
    parser.add_argument('--skip_eval', action='store_true', help='Only load existing results and compare (no eval runs).')
    parser.add_argument('-od', '--output_dir', type=str, default='eval_results', help='Base dir for eval_results.')
    parser.add_argument('-odc', '--comparison_output_dir', type=str, default=None, help='Dir for comparison JSON (default: eval_results/comparison).')
    args = parser.parse_args()
    if isinstance(args.editor_ckpt_path, str) and args.editor_ckpt_path.strip().lower() in {'none', 'null', ''}:
        args.editor_ckpt_path = None
    if args.ar_ckpt is not None and args.ar_ckpt.strip().lower() in {'none', 'null', ''}:
        args.ar_ckpt = None
    if args.ar_ckpt is None:
        args.ar_ckpt = args.editor_ckpt_path
    return args


def _eval_result_path(base_dir: str, postfix: str, sequential_edit_n: int, seed: int) -> str:
    """Path to mean_results.json for a given run (VLKEB-{postfix}, with optional seed in filename)."""
    editor_name = 'liveedit'
    model_name = get_full_model_name('blip2')
    eval_name = 'VLKEB-%s' % postfix
    save_dir = os.path.join(base_dir, editor_name, model_name, eval_name, 'sequential_edit_%s' % sequential_edit_n)
    if seed is not None:
        fname = 'seed_%s_mean_results.json' % seed
    else:
        fname = 'mean_results.json'
    return os.path.join(save_dir, fname)


def _run_eval(device: str, ckpt: str, ar_mode: bool, postfix: str, data_sample_n: int,
              sequential_edit_n: int, eval_seed: int, chunk_size: int, max_chunks, output_dir: str,
              vlkeb_eval_json: str = None):
    """Run one eval leg (baseline or AR) and save to VLKEB-{postfix}."""
    editor = load_vllm_editor('liveedit', 'blip2', device, None, ckpt, False)
    editor.ar_mode = ar_mode
    editor.chunk_size = chunk_size
    editor.max_chunks = max_chunks

    from dataset.vllm import VLKEB
    data_path = vlkeb_eval_json or os.path.join(ROOT_PATH, 'data/VLKEB/eval.json')
    if data_path and not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_PATH, data_path)
    # Use same image root as train (mmkb_images) so image paths in eval.json resolve
    img_root_dir = os.path.join(ROOT_PATH, 'data/VLKEB/VLKEB_images/mmkb_images')
    eval_data = VLKEB(data_path, img_root_dir, data_sample_n)

    evaluation_name = 'VLKEB-%s' % postfix
    ev = VLLMEditorEvaluation(editor, eval_data, evaluation_name, output_dir, ar_chunk_size=chunk_size)
    ev.evaluate_sequential_edit(sequential_edit_n, True, eval_seed)
    return _eval_result_path(output_dir, postfix, sequential_edit_n, eval_seed)


def _load_total_mean(path: str) -> dict:
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('total_mean', data)


def _compare_metrics(base: dict, ar: dict) -> dict:
    """Build comparison dict: keys with baseline vs AR and delta where applicable."""
    out = {'reliability': {}, 'generality': {}, 'locality': {}, 'length_bin': {}, 'validation': {}}
    for top in ('reliability', 'generality', 'locality'):
        if top not in base or top not in ar:
            continue
        out[top] = {}
        for k in set(base[top].keys()) | set(ar[top].keys()):
            bv = base[top].get(k)
            av = ar[top].get(k)
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                out[top][k] = {'baseline': bv, 'ar': av, 'delta': round(av - bv, 4)}
            elif isinstance(bv, dict) and isinstance(av, dict):
                out[top][k] = {}
                for sk in set(bv.keys()) | set(av.keys()):
                    bsv = bv.get(sk)
                    asv = av.get(sk)
                    if isinstance(bsv, (int, float)) and isinstance(asv, (int, float)):
                        out[top][k][sk] = {'baseline': bsv, 'ar': asv, 'delta': round(asv - bsv, 4)}
    if 'length_bin' in base and 'length_bin' in ar:
        for bin_name in ('<=16', '17-32', '33-64', '65+'):
            if bin_name not in base['length_bin'] or bin_name not in ar['length_bin']:
                continue
            bb = base['length_bin'][bin_name]
            ab = ar['length_bin'][bin_name]
            out['length_bin'][bin_name] = {
                'count_baseline': bb.get('count'), 'count_ar': ab.get('count'),
                'em': {'baseline': bb.get('em'), 'ar': ab.get('em'), 'delta': round((ab.get('em') or 0) - (bb.get('em') or 0), 4)},
                'acc': {'baseline': bb.get('acc'), 'ar': ab.get('acc'), 'delta': round((ab.get('acc') or 0) - (bb.get('acc') or 0), 4)},
                'chunk_em': {'baseline': bb.get('chunk_em'), 'ar': ab.get('chunk_em'), 'delta': round((ab.get('chunk_em') or 0) - (bb.get('chunk_em') or 0), 4)},
                'chunk_f1': {'baseline': bb.get('chunk_f1'), 'ar': ab.get('chunk_f1'), 'delta': round((ab.get('chunk_f1') or 0) - (bb.get('chunk_f1') or 0), 4)},
            }
    # Validation (plan: long-form gain, locality bounded)
    long_bin = out['length_bin'].get('65+', {})
    em_delta_long = long_bin.get('em', {}).get('delta')
    acc_delta_long = long_bin.get('acc', {}).get('delta')
    out['validation']['long_form_65+_em_delta'] = em_delta_long
    out['validation']['long_form_65+_acc_delta'] = acc_delta_long
    out['validation']['long_form_improvement'] = (em_delta_long is not None and em_delta_long > 0) or (acc_delta_long is not None and acc_delta_long > 0)

    loc_drop = None
    if 'locality' in base and 'locality' in ar:
        for sub in base['locality']:
            if sub in ar['locality'] and 'acc' in base['locality'][sub] and 'acc' in ar['locality'][sub]:
                d = ar['locality'][sub]['acc'] - base['locality'][sub]['acc']
                if loc_drop is None or d < loc_drop:
                    loc_drop = d
    out['validation']['locality_min_delta'] = round(loc_drop, 4) if loc_drop is not None else None
    out['validation']['locality_bounded'] = loc_drop is None or loc_drop >= -0.05  # allow up to 5% drop

    return out


def main():
    t0 = time.time()
    cfg = get_attr()
    if not cfg.skip_eval and not cfg.run_baseline_only and not cfg.run_ar_only:
        if cfg.editor_ckpt_path is None:
            print('No checkpoint provided; running with untrained editor (comparison of inference paths).')

    base_path = _eval_result_path(cfg.output_dir, 'baseline', cfg.sequential_edit_n, cfg.eval_seed)
    ar_path = _eval_result_path(cfg.output_dir, 'ar', cfg.sequential_edit_n, cfg.eval_seed)

    if not cfg.skip_eval:
        if not cfg.run_ar_only:
            print('Running baseline (ar_mode=False)...')
            _run_eval(cfg.device, cfg.editor_ckpt_path, False, 'baseline',
                      cfg.data_sample_n, cfg.sequential_edit_n, cfg.eval_seed,
                      cfg.chunk_size, cfg.max_chunks, cfg.output_dir,
                      getattr(cfg, 'vlkeb_eval_json', None))
        if not cfg.run_baseline_only:
            ckpt = cfg.ar_ckpt or cfg.editor_ckpt_path
            print('Running AR (ar_mode=True)...')
            _run_eval(cfg.device, ckpt, True, 'ar',
                      cfg.data_sample_n, cfg.sequential_edit_n, cfg.eval_seed,
                      cfg.chunk_size, cfg.max_chunks, cfg.output_dir,
                      getattr(cfg, 'vlkeb_eval_json', None))

    base_mean = _load_total_mean(base_path)
    ar_mean = _load_total_mean(ar_path)
    if base_mean is None:
        print('Baseline results not found at %s' % base_path)
        sys.exit(1)
    if ar_mean is None:
        print('AR results not found at %s' % ar_path)
        sys.exit(1)

    comparison = _compare_metrics(base_mean, ar_mean)
    elapsed = time.time() - t0
    comparison['config'] = {
        'data_sample_n': cfg.data_sample_n,
        'sequential_edit_n': cfg.sequential_edit_n,
        'eval_seed': cfg.eval_seed,
        'vlkeb_eval_json': getattr(cfg, 'vlkeb_eval_json', None),
        'baseline_ckpt': cfg.editor_ckpt_path,
        'ar_ckpt': cfg.ar_ckpt,
        'baseline_path': base_path,
        'ar_path': ar_path,
        'runtime_seconds': round(elapsed, 2),
        'runtime_minutes': round(elapsed / 60, 2),
    }

    # Print summary
    print('\n--- LiveEdit vs AR-LiveEdit comparison (VLKEB, seed=%s) ---' % cfg.eval_seed)
    print('Reliability: acc baseline=%.4f ar=%.4f delta=%.4f' % (
        base_mean['reliability'].get('acc', 0), ar_mean['reliability'].get('acc', 0),
        comparison['reliability'].get('acc', {}).get('delta', 0)))
    print('Reliability: em  baseline=%.4f ar=%.4f delta=%.4f' % (
        base_mean['reliability'].get('em', 0), ar_mean['reliability'].get('em', 0),
        comparison['reliability'].get('em', {}).get('delta', 0)))
    print('Length bin 65+: em delta=%.4f  acc delta=%.4f' % (
        comparison['validation'].get('long_form_65+_em_delta') or 0,
        comparison['validation'].get('long_form_65+_acc_delta') or 0))
    print('Locality min delta: %s  (bounded: %s)' % (
        comparison['validation'].get('locality_min_delta'),
        comparison['validation'].get('locality_bounded')))
    print('Long-form improvement: %s' % comparison['validation'].get('long_form_improvement'))

    out_dir = cfg.comparison_output_dir or os.path.join(cfg.output_dir, 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'liveedit_vs_ar_seed%s.json' % cfg.eval_seed)
    with open(out_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print('Comparison saved to %s' % out_path)
    print('[Timer] Total elapsed: %.2f s (%.2f min)' % (elapsed, elapsed / 60))

    if not comparison['validation'].get('locality_bounded', True):
        print('Warning: Locality drop exceeds 5% threshold.')
    if not comparison['validation'].get('long_form_improvement', False):
        print('Note: Long-form (65+) did not show improvement in this run.')


if __name__ == '__main__':
    main()
