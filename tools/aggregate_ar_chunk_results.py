"""
Aggregate AR eval runs across chunk_size settings into one comparison JSON + markdown table.

Usage (after running test_vllm_edit.py or run_ar_ablation.py for each chunk size):

  python tools/aggregate_ar_chunk_results.py --manifest tools/chunk_sweep_manifest.example.json

Manifest format:
  [
    {"chunk_size": 8,  "results_path": "eval_results/.../seed_42_mean_results.json"},
    {"chunk_size": 16, "results_path": "eval_results/.../seed_42_mean_results.json"}
  ]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def _load_total_mean(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('total_mean', data)


def _row_from_total_mean(total_mean: Dict[str, Any]) -> Dict[str, Any]:
    rel = total_mean.get('reliability', {})
    gen = total_mean.get('generality', {})
    loc = total_mean.get('locality', {})
    bin65 = total_mean.get('length_bin', {}).get('65+', {})
    loc_accs = [
        v.get('acc') for v in loc.values()
        if isinstance(v, dict) and v.get('acc') is not None
    ]
    return {
        'reliability_acc': rel.get('acc'),
        'reliability_em': rel.get('em'),
        'reliability_chunk_em': rel.get('chunk_em'),
        'reliability_chunk_f1': rel.get('chunk_f1'),
        'target_token_count': rel.get('target_token_count'),
        'generality_text_rephrase_acc': gen.get('text_rephrase', {}).get('acc'),
        'generality_image_rephrase_acc': gen.get('image_rephrase', {}).get('acc'),
        'locality_text_acc': loc.get('text_loc', {}).get('acc'),
        'locality_image_acc': loc.get('image_loc', {}).get('acc'),
        'locality_acc_min': min(loc_accs) if loc_accs else None,
        'bin_65p_count': bin65.get('count'),
        'bin_65p_acc': bin65.get('acc'),
        'bin_65p_em': bin65.get('em'),
        'bin_65p_chunk_em': bin65.get('chunk_em'),
        'bin_65p_chunk_f1': bin65.get('chunk_f1'),
        'total_edit_n': total_mean.get('total_edit_n'),
    }


def build_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    table = []
    for e in entries:
        total_mean = e.get('total_mean')
        if total_mean is None:
            continue
        row = _row_from_total_mean(total_mean)
        row['chunk_size'] = e['chunk_size']
        row['results_path'] = e.get('results_path')
        table.append(row)

    def score(r):
        return (
            r.get('bin_65p_chunk_em') or 0.0,
            r.get('bin_65p_acc') or 0.0,
            r.get('reliability_acc') or 0.0,
            r.get('locality_acc_min') if r.get('locality_acc_min') is not None else 1.0,
        )

    ranked = sorted(table, key=score, reverse=True)
    best = ranked[0] if ranked else None
    return {
        'chunk_size_table': table,
        'best_chunk_size': best['chunk_size'] if best else None,
        'recommendation': (
            'chunk_size=%s (highest 65+ chunk EM among provided runs)' % best['chunk_size']
            if best else 'N/A'
        ),
    }


def _pct(x: Optional[float]) -> str:
    if x is None:
        return 'N/A'
    return '%.2f%%' % (100.0 * x)


def write_markdown(report: Dict[str, Any], out_path: str) -> None:
    lines = [
        '# AR chunk size sweep (VLKEB-long)',
        '',
        '**Recommendation:** %s' % report.get('recommendation', 'N/A'),
        '',
        '| chunk_size | rel acc | chunk EM | chunk F1 | 65+ acc | 65+ chunk EM | text rephrase | image rephrase | locality min | n edits |',
        '|------------|---------|----------|----------|---------|--------------|---------------|----------------|--------------|---------|',
    ]
    for r in report.get('chunk_size_table', []):
        lines.append(
            '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                r.get('chunk_size'),
                _pct(r.get('reliability_acc')),
                _pct(r.get('reliability_chunk_em')),
                _pct(r.get('reliability_chunk_f1')),
                _pct(r.get('bin_65p_acc')),
                _pct(r.get('bin_65p_chunk_em')),
                _pct(r.get('generality_text_rephrase_acc')),
                _pct(r.get('generality_image_rephrase_acc')),
                _pct(r.get('locality_acc_min')),
                r.get('total_edit_n'),
            )
        )
    lines.append('')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True, help='JSON list of {chunk_size, results_path}')
    parser.add_argument('-o', '--output_json', type=str, default='eval_results/comparison/ar_chunk_size_sweep.json')
    parser.add_argument('--output_md', type=str, default='doc/THESIS_AR_CHUNK_SIZE_SWEEP.md')
    args = parser.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    entries = []
    missing = []
    for item in manifest:
        path = item['results_path']
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        total_mean = _load_total_mean(path)
        if total_mean is None:
            missing.append((item.get('chunk_size'), path))
            continue
        entries.append({
            'chunk_size': item['chunk_size'],
            'results_path': path,
            'total_mean': total_mean,
        })

    if missing:
        print('Missing results:')
        for cs, p in missing:
            print('  chunk_size=%s  %s' % (cs, p))

    report = build_report(entries)
    report['missing'] = [{'chunk_size': cs, 'results_path': p} for cs, p in missing]

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('Wrote %s' % args.output_json)

    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) or '.', exist_ok=True)
        write_markdown(report, args.output_md)
        print('Wrote %s' % args.output_md)

    if report.get('chunk_size_table'):
        print('\n--- chunk_size sweep ---')
        print('chunk_size | rel_acc | chunk_em | 65+_acc | 65+_chunk_em | loc_min')
        for r in report['chunk_size_table']:
            print('%9s | %7.4f | %8.4f | %7.4f | %12.4f | %s' % (
                r.get('chunk_size'),
                r.get('reliability_acc') or 0,
                r.get('reliability_chunk_em') or 0,
                r.get('bin_65p_acc') or 0,
                r.get('bin_65p_chunk_em') or 0,
                '%.4f' % r['locality_acc_min'] if r.get('locality_acc_min') is not None else 'N/A',
            ))
        print('\nRecommendation: %s' % report.get('recommendation'))


if __name__ == '__main__':
    main()
