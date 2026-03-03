#%%
"""
Run long-form comparison with a small dataset: train baseline and AR on small VLKEB,
then run LiveEdit vs AR-LiveEdit comparison so length bins (including 33-64, 65+) can be compared.
Uses same seed for train and eval for reproducibility.
"""
from utils.GLOBAL import ROOT_PATH
from utils import get_full_model_name
import os
import sys
import glob
import subprocess
import argparse


def get_attr():
    parser = argparse.ArgumentParser(description='Train baseline + AR on small VLKEB, then run long-form comparison.')
    parser.add_argument('-dvc', '--device', type=str, default='cuda:0', help='CUDA device.')
    parser.add_argument('-tn', '--train_n', type=int, default=40, help='Training samples (VLKEB).')
    parser.add_argument('-bs', '--batch_size', type=int, default=2, help='Train batch size.')
    parser.add_argument('-eps', '--epochs', type=int, default=2, help='Train epochs.')
    parser.add_argument('-sci', '--save_ckpt_per_i', type=int, default=30, help='Save checkpoint every N iters.')
    parser.add_argument('-dsn', '--eval_sample_n', type=int, default=60, help='Eval samples for comparison.')
    parser.add_argument('-sen', '--sequential_edit_n', type=int, default=20, help='Edits per sequential batch.')
    parser.add_argument('-seed', '--seed', type=int, default=42, help='Random seed for train and eval.')
    parser.add_argument('--skip_train', action='store_true', help='Skip training; use existing longform_baseline / longform_ar checkpoints.')
    parser.add_argument('--baseline_ckpt', type=str, default=None, help='Override: path to baseline checkpoint.')
    parser.add_argument('--ar_ckpt', type=str, default=None, help='Override: path to AR checkpoint.')
    return parser.parse_args()


def _find_latest_ckpt_in_dir(ckpt_dir: str) -> str:
    """Return path to one checkpoint in dir (prefer latest by mtime)."""
    pattern = os.path.join(ckpt_dir, '*.pt')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _find_run_ckpt(records_base: str, prefix: str) -> str:
    """Find latest checkpoint from a run whose name starts with prefix."""
    editor_name = 'liveedit'
    model_name = get_full_model_name('blip2')
    parent = os.path.join(records_base, editor_name, model_name)
    if not os.path.isdir(parent):
        return None
    best_ckpt = None
    best_mtime = 0
    for run_dir in os.listdir(parent):
        if not run_dir.startswith(prefix):
            continue
        ckpt_dir = os.path.join(parent, run_dir, 'checkpoints')
        ckpt = _find_latest_ckpt_in_dir(ckpt_dir)
        if ckpt and os.path.getmtime(ckpt) > best_mtime:
            best_mtime = os.path.getmtime(ckpt)
            best_ckpt = ckpt
    return best_ckpt


def main():
    cfg = get_attr()
    # Records are written under cwd when training; cwd is repo root (LiveEdit)
    root = os.path.dirname(os.path.abspath(__file__))
    records_dir = os.path.join(root, 'records')

    baseline_ckpt = cfg.baseline_ckpt
    ar_ckpt = cfg.ar_ckpt

    if not cfg.skip_train:
        # 1. Train baseline (no AR)
        print('=== Training baseline (ar_mode=False) on small VLKEB ===')
        cmd_baseline = [
            sys.executable, '-W', 'ignore', 'train_vllm_editor.py',
            '-en', 'LiveEdit', '-mn', 'blip2', '-dna', 'VLKEB', '-bs', str(cfg.batch_size),
            '-dvc', cfg.device, '-dn', str(cfg.train_n), '-eps', str(cfg.epochs),
            '-sci', str(cfg.save_ckpt_per_i), '-tnp', 'longform_baseline', '-rs', str(cfg.seed),
        ]
        r = subprocess.run(cmd_baseline, cwd=root)
        if r.returncode != 0:
            print('Baseline training failed.')
            sys.exit(1)

        # 2. Train AR
        print('=== Training AR (ar_mode=True) on small VLKEB ===')
        cmd_ar = [
            sys.executable, '-W', 'ignore', 'train_vllm_editor.py',
            '-en', 'LiveEdit', '-mn', 'blip2', '-dna', 'VLKEB', '-bs', str(cfg.batch_size),
            '-dvc', cfg.device, '-dn', str(cfg.train_n), '-eps', str(cfg.epochs),
            '-sci', str(cfg.save_ckpt_per_i), '-tnp', 'longform_ar', '--ar_mode', '-rs', str(cfg.seed),
        ]
        r = subprocess.run(cmd_ar, cwd=root)
        if r.returncode != 0:
            print('AR training failed.')
            sys.exit(1)
    else:
        print('Skipping training (--skip_train).')

    if not baseline_ckpt:
        baseline_ckpt = _find_run_ckpt(records_dir, 'longform_baseline')
    if not ar_ckpt:
        ar_ckpt = _find_run_ckpt(records_dir, 'longform_ar')

    if not baseline_ckpt or not os.path.isfile(baseline_ckpt):
        print('Baseline checkpoint not found. Expected a run with prefix longform_baseline.')
        sys.exit(1)
    if not ar_ckpt or not os.path.isfile(ar_ckpt):
        print('AR checkpoint not found. Expected a run with prefix longform_ar.')
        sys.exit(1)

    print('Baseline checkpoint:', baseline_ckpt)
    print('AR checkpoint:', ar_ckpt)

    # 3. Run comparison
    print('=== Running LiveEdit vs AR-LiveEdit comparison (long-form eval) ===')
    cmd_compare = [
        sys.executable, '-W', 'ignore', 'run_liveedit_ar_comparison.py',
        '-dvc', cfg.device, '-ckpt', baseline_ckpt, '--ar_ckpt', ar_ckpt,
        '-dsn', str(cfg.eval_sample_n), '-sen', str(cfg.sequential_edit_n),
        '-seed', str(cfg.seed),
    ]
    r = subprocess.run(cmd_compare, cwd=root)
    if r.returncode != 0:
        print('Comparison failed.')
        sys.exit(1)
    print('Done. Check eval_results/comparison/liveedit_vs_ar_seed%s.json for length_bin (<=16, 17-32, 33-64, 65+).' % cfg.seed)


if __name__ == '__main__':
    main()
