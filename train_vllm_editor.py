#%%
from utils.GLOBAL import ROOT_PATH
from utils import load_vllm_editor
import os, argparse, time, json


def get_attr():
    def parse_lkpt(value:str):
        if value.lower() == 'none':
            return None
        return value
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument('-en', '--editor_name', type=str, help='Editor name: LiveEdit, FT_VL...', required=True)
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: llava...', required=True)
    parser.add_argument('-dna', '--data_name', type=str, help = 'Train dataset, including EVQA, EIC.', required = True)
    parser.add_argument('-bs', '--batch_size', type=int, help = 'Train dataset sample number.', required = True)
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    # other settings
    parser.add_argument('-dn', '--data_n', type=int, default=None, help = 'Train dataset sample number.')
    parser.add_argument('-lkpt', '--load_ckpt_path', type=parse_lkpt, default = None, help='For Editors that needs training.')
    parser.add_argument('-edvc', '--extra_devices', type=int, nargs='+', default = [0], help='Extra CUDA devices, default empty.')
    parser.add_argument('-eps', '--epochs', type=int, default=1000, help = 'Train epochs.')
    parser.add_argument('-tnp', '--train_name_prefix', type=str, default=None, help = 'Train name prefix.')
    parser.add_argument('-sci', '--save_ckpt_per_i', type=int, default=1000, help = 'Save checkpoint per iteraions.')
    parser.add_argument('-lpi', '--log_per_i', type=int, default=1, help = 'Log per iteraions.')
    parser.add_argument('-ea', '--ema_alpha', type=float, default=0.1, help = 'EMA loss alpha.')
    parser.add_argument('-rs', '--random_seed', type=int, default=None, help = 'Random seed.')
    parser.add_argument('-dbs', '--data_buffer_size', type=int, default=4, help = 'Buffer size of data generator.')
    parser.add_argument(
        '-dpath',
        '--vlkeb_data_json',
        type=str,
        default=None,
        help='VLKEB only: override path to training JSON (default: data/VLKEB/train.json under ROOT_PATH).',
    )
    # AR-LiveEdit (no behavior change when ar_mode=False)
    parser.add_argument('--ar_mode', action='store_true', help='Enable autoregressive chunk-wise edit mode.')
    parser.add_argument('--chunk_size', type=int, default=16, help='Target chunk size in tokens for AR mode.')
    parser.add_argument('--max_chunks', type=int, default=None, help='Optional max chunks per target in AR mode (default: unlimited).')
    parser.add_argument('--patch_grounding', action='store_true', help='Step 6: enable PatchGroundingModule training.')
    parser.add_argument('--patch_encoder', type=str, default='dinov2', choices=['dinov2', 'eva_patches'])
    parser.add_argument('--adversarial_grounding', action='store_true', help='VG-Edit AVBL: swap/hard-neg binding loss.')
    parser.add_argument('--adv_grounding_lambda', type=float, default=1.0)
    parser.add_argument('--adv_grounding_neg_weight', type=float, default=1.0,
                        help='Weight on AVBL negative (swap/hard-neg) terms vs positive.')
    parser.add_argument('--avbl_hinge_tau', type=float, default=0.0,
                        help='Hinge: apply negative AVBL only when expert_mass > tau (0=off).')
    parser.add_argument('--avbl_contrastive', action='store_true',
                        help='Add contrastive loss penalizing max wrong-expert prob on adversarial probes.')
    parser.add_argument('--avbl_contrastive_margin', type=float, default=0.1)
    parser.add_argument('--avbl_no_targeted_false_cross', action='store_true',
                        help='Disable E1 targeted top-wrong-expert loss.')
    parser.add_argument('--avbl_rel_anchor_min', type=float, default=0.5,
                        help='E3: scale negative AVBL by min(1, own_prob/anchor_min).')
    parser.add_argument('--avbl_retrieval_margin', type=float, default=0.0,
                        help='Retrieval-margin hinge on pre-softmax vis_sim (0=off). Try 0.3.')
    parser.add_argument('--avbl_dino_neighbor', action='store_true',
                        help='Mine DINO-nearest wrong batch expert and suppress it under adv images.')
    parser.add_argument('--avbl_neighbor_weight', type=float, default=2.0,
                        help='Weight on DINO-mined neighbor suppression loss.')
    parser.add_argument('--avbl_refuse_lambda', type=float, default=0.0,
                        help='Token refuse loss weight on adv images (0=off). Try 0.2.')
    parser.add_argument('--avbl_refuse_tau', type=float, default=-1.0,
                        help='Refuse hinge: penalize if mean logP(target|adv) > tau.')
    parser.add_argument('--avbl_pool_buffer', type=int, default=0,
                        help='Phase D: cross-batch buffer size for DINO neighbor mining (0=off). Try 64.')
    parser.add_argument('--fused_expert_key', action='store_true',
                        help='LEGACY DEK: QFormer+patch embed fuse (collapses rel — prefer --dek_dual_gate).')
    parser.add_argument('--dek_dual_gate', action='store_true',
                        help='DEK: retrieve only if visual gate AND prompt-key agreement.')
    parser.add_argument('--dek_text_margin', type=float, default=0.0,
                        help='DEK dual-gate: allow txt_sim within margin of global max (0=argmax only).')
    parser.add_argument('--dek_text_alpha', type=float, default=1.0,
                        help='DEK: weight of txt_sim added into AVBL binding logits.')
    parser.add_argument('--pool_dedup', action='store_true',
                        help='DEK: DINO near-duplicate pool hygiene on edit.')
    parser.add_argument('--pool_dedup_threshold', type=float, default=0.95,
                        help='DINO cosine threshold for pool dedup.')
    parser.add_argument('--pool_dedup_mode', type=str, default='merge', choices=['merge', 'skip'],
                        help='merge=overwrite near-dup slot; skip=drop new edit.')
    parser.add_argument('--records_dir', type=str, default=None, help='Override records root (e.g. records/meeting_validation/step6).')
    parser.add_argument('--resume', action='store_true', help='Resume training from -lkpt (keep epoch/i, reuse run dir).')
    parser.add_argument('--resume_run_name', type=str, default=None,
                        help='Existing run folder name under records/liveedit/<model>/ (with --resume).')
    parser.add_argument(
        '--keep_latest_ckpts', type=int, default=None,
        help='After each save, keep only the newest N checkpoint files (e.g. 3). Default: keep all.',
    )
    args = parser.parse_args()
    return args


class cfg:
    editor_name = 'LiveEdit'
    edit_model_name = 'blip2'
    data_name = 'EVQA'
    batch_size = 4
    device = 'cuda:0'
    data_n = None
    load_ckpt_path = None
    extra_devices = [1]
    epochs = 1000
    train_name_prefix = 'testttt'
    save_ckpt_per_i = 1000
    log_per_i = 1
    ema_alpha = 0.1
    random_seed = 1
    data_buffer_size = 4
    # AR-LiveEdit
    ar_mode = False
    chunk_size = 16
    max_chunks = None


if __name__ == '__main__':
    t0 = time.time()
    cfg = get_attr()
    cfg.data_name = cfg.data_name.upper()
    # load editor
    editor = load_vllm_editor(cfg.editor_name, cfg.edit_model_name, cfg.device, cfg.extra_devices, None, True)
    # load data
    if cfg.data_name == 'EVQA':
        data_path = getattr(cfg, 'vlkeb_data_json', None) or os.path.join(
            ROOT_PATH, 'data/easy-edit-mm/vqa/vqa_train.json')
        if data_path and not os.path.isabs(data_path):
            data_path = os.path.join(ROOT_PATH, data_path)
        img_root_dir = os.path.join(ROOT_PATH, 'data/easy-edit-mm/images')
        if getattr(cfg, 'adversarial_grounding', False) and 'grounding' in os.path.basename(data_path):
            from dataset.vllm_grounding import GroundingEVQA
            train_data = GroundingEVQA(data_path, img_root_dir, cfg.data_n)
        else:
            from dataset.vllm import EVQA
            train_data = EVQA(data_path, img_root_dir, cfg.data_n)
    elif cfg.data_name == 'EIC':
        from dataset.vllm import EIC
        data_path = os.path.join(ROOT_PATH, 'data/easy-edit-mm/caption/caption_train_edit.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/easy-edit-mm/images')
        train_data = EIC(data_path, img_root_dir, cfg.data_n)
    elif cfg.data_name == 'VLKEB':
        from dataset.vllm import VLKEB
        data_path = getattr(cfg, 'vlkeb_data_json', None) or os.path.join(ROOT_PATH, 'data/VLKEB/train.json')
        if data_path and not os.path.isabs(data_path):
            data_path = os.path.join(ROOT_PATH, data_path)
        # VLKEB images are stored under data/VLKEB/VLKEB_images/mmkb_images in this repo snapshot.
        img_root_dir = os.path.join(ROOT_PATH, 'data/VLKEB/VLKEB_images/mmkb_images')
        if getattr(cfg, 'adversarial_grounding', False) and 'grounding' in os.path.basename(data_path):
            from dataset.vllm_grounding import GroundingVLKEB
            train_data = GroundingVLKEB(data_path, img_root_dir, cfg.data_n)
        else:
            train_data = VLKEB(data_path, img_root_dir, cfg.data_n)
    # AR-LiveEdit: pass AR config into editor for batch organization (Task 3)
    editor.ar_mode = getattr(cfg, 'ar_mode', False)
    editor.chunk_size = getattr(cfg, 'chunk_size', 16)
    editor.max_chunks = getattr(cfg, 'max_chunks', None)
    if getattr(cfg, 'patch_grounding', False):
        editor.enable_patch_grounding(
            encoder=getattr(cfg, 'patch_encoder', 'dinov2'),
            expert_gen=True,
        )
    if getattr(cfg, 'adversarial_grounding', False):
        editor.enable_adversarial_grounding(
            lambda_adv=getattr(cfg, 'adv_grounding_lambda', 1.0),
            neg_weight=getattr(cfg, 'adv_grounding_neg_weight', 1.0),
            hinge_tau=getattr(cfg, 'avbl_hinge_tau', 0.0),
            use_contrastive=getattr(cfg, 'avbl_contrastive', False),
            contrastive_margin=getattr(cfg, 'avbl_contrastive_margin', 0.1),
            targeted_false_cross=not getattr(cfg, 'avbl_no_targeted_false_cross', False),
            rel_anchor_min=getattr(cfg, 'avbl_rel_anchor_min', 0.5),
            retrieval_margin=getattr(cfg, 'avbl_retrieval_margin', 0.0),
            dino_neighbor_mine=getattr(cfg, 'avbl_dino_neighbor', False),
            neighbor_weight=getattr(cfg, 'avbl_neighbor_weight', 2.0),
            refuse_lambda=getattr(cfg, 'avbl_refuse_lambda', 0.0),
            refuse_tau=getattr(cfg, 'avbl_refuse_tau', -1.0),
            pool_buffer_size=getattr(cfg, 'avbl_pool_buffer', 0),
        )
    if (
        getattr(cfg, 'fused_expert_key', False)
        or getattr(cfg, 'pool_dedup', False)
        or getattr(cfg, 'dek_dual_gate', False)
    ):
        editor.enable_dek(
            fused_expert_key=getattr(cfg, 'fused_expert_key', False),
            pool_dedup=getattr(cfg, 'pool_dedup', False),
            pool_dedup_threshold=getattr(cfg, 'pool_dedup_threshold', 0.95),
            dual_gate=getattr(cfg, 'dek_dual_gate', False),
            text_margin=getattr(cfg, 'dek_text_margin', 0.0),
            text_alpha=getattr(cfg, 'dek_text_alpha', 1.0),
            pool_dedup_mode=getattr(cfg, 'pool_dedup_mode', 'merge'),
            encoder=getattr(cfg, 'patch_encoder', 'dinov2'),
        )
    records_dir = getattr(cfg, 'records_dir', None) or 'records'
    # initialize and train
    editor.train_init(train_data, cfg.batch_size, train_name_prefix = cfg.train_name_prefix,
        load_ckpt_path = cfg.load_ckpt_path, save_ckpt_per_i = cfg.save_ckpt_per_i, 
        log_per_i = cfg.log_per_i, ema_alpha = cfg.ema_alpha, random_seed = cfg.random_seed,
        data_buffer_size = cfg.data_buffer_size, records_dir=records_dir,
        resume=getattr(cfg, 'resume', False),
        resume_run_name=getattr(cfg, 'resume_run_name', None),
        keep_latest_ckpts=getattr(cfg, 'keep_latest_ckpts', None))
    editor.train(cfg.epochs)
    elapsed = time.time() - t0
    runtime = {"runtime_seconds": round(elapsed, 2), "runtime_minutes": round(elapsed / 60, 2)}
    run_dir = os.path.dirname(editor.save_ckpt_dir)
    runtime_path = os.path.join(run_dir, "runtime.json")
    with open(runtime_path, "w") as f:
        json.dump(runtime, f, indent=2)
    print("[Timer] Total elapsed: %.2f s (%.2f min). Saved to %s" % (elapsed, elapsed / 60, runtime_path))
