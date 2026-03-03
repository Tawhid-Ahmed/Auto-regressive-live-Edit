#%%
"""
Task 9: Small ablation set for AR-LiveEdit (chunk_size, routing gate, max_chunks).
Runs AR eval for each combination, then produces one compact ablation table with recommendation.
"""
from utils import get_full_model_name, load_vllm_editor
from evaluation.vllm_editor_eval import VLLMEditorEvaluation
from utils.GLOBAL import ROOT_PATH
import os
import sys
import json
import argparse
import time
from typing import Optional, List, Dict, Any


def get_attr():
    parser = argparse.ArgumentParser(description='AR-LiveEdit ablation: chunk_size, routing_gate, max_chunks.')
    parser.add_argument('-dvc', '--device', type=str, default='cuda:0', help='CUDA device.')
    parser.add_argument('-ckpt', '--editor_ckpt_path', type=str, default=None, help='Checkpoint path.')
    parser.add_argument('-dsn', '--data_sample_n', type=int, default=100, help='VLKEB eval sample count (small for ablation).')
    parser.add_argument('-sen', '--sequential_edit_n', type=int, default=25, help='Edits per sequential batch.')
    parser.add_argument('-seed', '--eval_seed', type=int, default=42, help='Fixed seed.')
    parser.add_argument('-od', '--output_dir', type=str, default='eval_results', help='Base dir for eval results.')
    parser.add_argument('--ablation_dir', type=str, default=None, help='Dir for ablation table (default: output_dir/ablation).')
    parser.add_argument('--skip_run', action='store_true', help='Only build table from existing results (no eval).')
    args = parser.parse_args()
    if getattr(args, 'editor_ckpt_path', None) and str(args.editor_ckpt_path).strip().lower() in {'none', 'null', ''}:
        args.editor_ckpt_path = None
    return args


# Ablation grid: chunk_size, routing_gate (use chunk-aware retrieval), max_chunks
CHUNK_SIZES = [8, 16, 32]
ROUTING_GATES = [True, False]   # True = gate on, False = gate off (full pool)
MAX_CHUNKS_OPTS = [None, 4]     # None = unlimited, 4 = capped


def _ablation_postfix(chunk_size: int, routing_gate: bool, max_chunks: Optional[int]) -> str:
    mc = 'unlim' if max_chunks is None else f'max{max_chunks}'
    gate = 'gate_on' if routing_gate else 'gate_off'
    return 'ar_ablation_cs%d_%s_%s' % (chunk_size, gate, mc)


def _eval_result_path(base_dir: str, postfix: str, sequential_edit_n: int, seed: int) -> str:
    editor_name = 'liveedit'
    model_name = get_full_model_name('blip2')
    eval_name = 'VLKEB-%s' % postfix
    save_dir = os.path.join(base_dir, editor_name, model_name, eval_name, 'sequential_edit_%s' % sequential_edit_n)
    fname = 'seed_%s_mean_results.json' % seed
    return os.path.join(save_dir, fname)


def _run_one_ablation(device: str, ckpt: str, chunk_size: int, routing_gate: bool, max_chunks: Optional[int],
                      data_sample_n: int, sequential_edit_n: int, eval_seed: int, output_dir: str) -> str:
    """Run one AR eval with given knobs; return path to mean_results.json."""
    editor = load_vllm_editor('liveedit', 'blip2', device, None, ckpt, False)
    editor.ar_mode = True
    editor.chunk_size = chunk_size
    editor.max_chunks = max_chunks
    editor.ar_use_routing_gate = routing_gate

    from dataset.vllm import VLKEB
    data_path = os.path.join(ROOT_PATH, 'data/VLKEB/eval.json')
    img_root_dir = os.path.join(ROOT_PATH, 'data/VLKEB/VLKEB_images/mmkb_images')
    eval_data = VLKEB(data_path, img_root_dir, data_sample_n)

    postfix = _ablation_postfix(chunk_size, routing_gate, max_chunks)
    evaluation_name = 'VLKEB-%s' % postfix
    ev = VLLMEditorEvaluation(editor, eval_data, evaluation_name, output_dir, ar_chunk_size=chunk_size)
    ev.evaluate_sequential_edit(sequential_edit_n, True, eval_seed)
    return _eval_result_path(output_dir, postfix, sequential_edit_n, eval_seed)


def _load_total_mean(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('total_mean', data)


def _extract_row(total_mean: Dict) -> Dict[str, Any]:
    """Extract key metrics for one run into a flat row."""
    row = {}
    for top in ('reliability', 'generality', 'locality'):
        if top not in total_mean:
            continue
        for k, v in total_mean[top].items():
            if isinstance(v, (int, float)):
                row['%s_%s' % (top, k)] = v
            elif isinstance(v, dict) and 'acc' in v:
                row['%s_%s_acc' % (top, k)] = v.get('acc')
    if 'length_bin' in total_mean:
        for bin_name in ('<=16', '17-32', '33-64', '65+'):
            if bin_name in total_mean['length_bin']:
                b = total_mean['length_bin'][bin_name]
                key = bin_name.replace('-', '_').replace('+', 'p')
                row['bin_%s_em' % key] = b.get('em')
                row['bin_%s_acc' % key] = b.get('acc')
                row['bin_%s_count' % key] = b.get('count')
    return row


def run_ablations(cfg) -> List[Dict[str, Any]]:
    """Run all ablation cells; return list of {config, path, row}."""
    results = []
    for chunk_size in CHUNK_SIZES:
        for routing_gate in ROUTING_GATES:
            for max_chunks in MAX_CHUNKS_OPTS:
                postfix = _ablation_postfix(chunk_size, routing_gate, max_chunks)
                path = _eval_result_path(cfg.output_dir, postfix, cfg.sequential_edit_n, cfg.eval_seed)
                if not cfg.skip_run:
                    print('Running: chunk_size=%s routing_gate=%s max_chunks=%s -> %s' % (
                        chunk_size, routing_gate, max_chunks, postfix))
                    path = _run_one_ablation(
                        cfg.device, cfg.editor_ckpt_path, chunk_size, routing_gate, max_chunks,
                        cfg.data_sample_n, cfg.sequential_edit_n, cfg.eval_seed, cfg.output_dir)
                total_mean = _load_total_mean(path)
                row = _extract_row(total_mean) if total_mean else {}
                results.append({
                    'chunk_size': chunk_size,
                    'routing_gate': routing_gate,
                    'max_chunks': max_chunks,
                    'postfix': postfix,
                    'path': path,
                    'total_mean': total_mean,
                    'row': row,
                })
    return results


def build_ablation_table(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build compact table and recommendation."""
    # Table: list of config + key metrics
    table = []
    for r in results:
        row = {
            'chunk_size': r['chunk_size'],
            'routing_gate': r['routing_gate'],
            'max_chunks': r['max_chunks'],
            'reliability_acc': r['row'].get('reliability_acc'),
            'reliability_em': r['row'].get('reliability_em'),
            'bin_65p_em': r['row'].get('bin_65p_em'),
            'bin_65p_acc': r['row'].get('bin_65p_acc'),
            'locality_acc_min': None,
        }
        # Min locality acc across locality sub-metrics
        loc_keys = [k for k in r['row'] if k.startswith('locality_') and k.endswith('_acc')]
        if loc_keys:
            row['locality_acc_min'] = min(r['row'][k] for k in loc_keys if r['row'][k] is not None)
        table.append(row)

    # Recommendation: best by long-form (65+) EM, then reliability, then locality bounded
    def score(entry):
        em65 = entry.get('bin_65p_em')
        acc65 = entry.get('bin_65p_acc')
        rel_acc = entry.get('reliability_acc')
        loc_min = entry.get('locality_acc_min')
        if em65 is None and acc65 is None:
            em65, acc65 = 0.0, 0.0
        em65 = em65 or 0.0
        acc65 = acc65 or 0.0
        rel_acc = rel_acc or 0.0
        loc_min = loc_min if loc_min is not None else 1.0
        return (em65, acc65, rel_acc, loc_min)

    ranked = sorted(table, key=score, reverse=True)
    best = ranked[0] if ranked else None
    recommendation = {
        'best_config': best,
        'recommendation': (
            'chunk_size=%s, routing_gate=%s, max_chunks=%s' % (
                best['chunk_size'], best['routing_gate'], best['max_chunks']
            ) if best else 'N/A'
        ),
        'note': 'Prefer long-form (65+) EM/acc then reliability then locality; gate_on often helps chunk-specific retrieval.',
    }

    return {
        'ablation_table': table,
        'recommendation': recommendation,
        'config': {
            'chunk_sizes': CHUNK_SIZES,
            'routing_gates': ROUTING_GATES,
            'max_chunks_opts': MAX_CHUNKS_OPTS,
        },
    }


def main():
    t0 = time.time()
    cfg = get_attr()
    if not cfg.skip_run and cfg.editor_ckpt_path is None:
        print('Warning: No checkpoint provided; running with untrained editor (ablation of inference path only).')

    results = run_ablations(cfg)
    report = build_ablation_table(results)
    elapsed = time.time() - t0
    report['runtime_seconds'] = round(elapsed, 2)
    report['runtime_minutes'] = round(elapsed / 60, 2)

    # Print compact table
    print('\n--- AR-LiveEdit ablation table ---')
    print('chunk_size | routing_gate | max_chunks | rel_acc  | rel_em   | bin_65+_em | bin_65+_acc | loc_min')
    print('-' * 95)
    for row in report['ablation_table']:
        ra = row.get('reliability_acc')
        re = row.get('reliability_em')
        e65 = row.get('bin_65p_em')
        a65 = row.get('bin_65p_acc')
        loc = row.get('locality_acc_min')
        print('%9s | %11s | %9s | %.4f   | %.4f   | %10s | %11s | %s' % (
            row['chunk_size'],
            row['routing_gate'],
            str(row['max_chunks']) if row['max_chunks'] is not None else 'unlim',
            ra if ra is not None else 0.0,
            re if re is not None else 0.0,
            '%.4f' % e65 if e65 is not None else 'N/A',
            '%.4f' % a65 if a65 is not None else 'N/A',
            '%.4f' % loc if loc is not None else 'N/A',
        ))
    print('\nRecommendation: %s' % report['recommendation']['recommendation'])
    print('Note: %s' % report['recommendation']['note'])

    out_dir = cfg.ablation_dir or os.path.join(cfg.output_dir, 'ablation')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ar_ablation_seed%s.json' % cfg.eval_seed)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print('Ablation report saved to %s' % out_path)
    print('[Timer] Total elapsed: %.2f s (%.2f min)' % (elapsed, elapsed / 60))


if __name__ == '__main__':
    main()
