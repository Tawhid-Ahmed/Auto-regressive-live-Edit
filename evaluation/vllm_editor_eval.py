from editor.vllm_editors.base import VLLMBaseEditor
from editor.vllms_for_edit.base import BaseVLLMForEdit
from dataset.vllm import BaseVLLMEditData
from typing import List, Dict, Union
from collections import defaultdict
from datetime import datetime
from copy import deepcopy
import torch, os, json
from tqdm import tqdm
from time import time
import numpy as np

# Length-bin boundaries for AR-specific metrics (plan: <=16, 17-32, 33-64, 65+)
LENGTH_BIN_EDGES = (16, 32, 64)  # bins: (0,16], (16,32], (32,64], (64,+inf)


def _length_bin_name(num_tokens: int) -> str:
    """Return bin name for a target token count."""
    if num_tokens <= LENGTH_BIN_EDGES[0]:
        return "<=16"
    if num_tokens <= LENGTH_BIN_EDGES[1]:
        return "17-32"
    if num_tokens <= LENGTH_BIN_EDGES[2]:
        return "33-64"
    return "65+"


def _chunk_em_f1(pre_y: torch.Tensor, label_ids: torch.Tensor, label_masks: torch.Tensor, chunk_size: int):
    """
    Compute per-chunk EM and token F1 over aligned prediction/label, then average.
    pre_y, label_ids, label_masks: [1, L]. chunk_size: max tokens per chunk.
    Returns (chunk_em, chunk_f1) in [0,1]; (0, 0) if no tokens.
    """
    L = label_ids.shape[1]
    n = label_masks.sum().item()
    if n == 0 or chunk_size <= 0:
        return 0.0, 0.0
    ems, f1s = [], []
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        m = label_masks[:, start:end]
        if m.sum() == 0:
            continue
        pred = pre_y[:, start:end]
        ref = label_ids[:, start:end]
        m_bool = m.bool() if m.dtype != torch.bool else m
        chunk_em = ((pred == ref) | ~m_bool).all().item()
        correct = ((pred == ref) * m).sum().item()
        chunk_acc = correct / m.sum().item()
        ems.append(float(chunk_em))
        f1s.append(float(chunk_acc))
    if not ems:
        return 0.0, 0.0
    return float(np.mean(ems)), float(np.mean(f1s))


class VLLMEditorEvaluation():
    def __init__(self, editor:VLLMBaseEditor, eval_data:BaseVLLMEditData, 
        evaluation_name = None, results_dir = 'eval_results', ar_chunk_size: int = 16) -> None:
        '''
        `results_dir` & `evaluation_name`: Used to create result directory.
            `evaluation_name` can be set as dataset name.
        `ar_chunk_size`: Chunk size for AR-specific chunk EM/F1 (default 16).
        '''
        self.editor = editor
        self.eval_data = eval_data
        self.ar_chunk_size = ar_chunk_size
        editor_name, model_name = editor.name_of_editor_and_model()
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        evaluation_name = evaluation_name if evaluation_name else t
        self.result_dir = os.path.join(results_dir, editor_name, model_name, evaluation_name)
        print('Evaluation results directory: ', self.result_dir)
        # self.eval_xyms = None

    def evaluate_single_edit(self):
        editor = self.editor
        print('Evaluating reliability, generality and locality for %s on %s with single editing.'
              %editor.name_of_editor_and_model())
        eval_data = deepcopy(self.eval_data.data_with_img)
        for ed in eval_data: # single edit, the number of requests must be 1.
            assert len(ed['requests']) == 1
        result_data = deepcopy(self.eval_data.data_with_img_path)
        tokenizer = editor.vllm.get_llm_tokenizer()
        editor.restore_to_original_model()  
        results = [] 
        for rd, ed in zip(tqdm(result_data, 'Evaluating'), eval_data):
            rd['reliability'] = rd.pop('requests') 
            rd['reliability'][0]['target'] = rd['reliability'][0].pop('target_new')
            # predict before edit for locality data
            for loc_name in ed['locality'].keys():
                for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                    (input_embeds, vt_range), label_ids, label_masks = editor.vllm.prompts_imgs_target_to_xym(
                        [edl['prompt']], [edl['image']], [edl['target']])
                    logits = editor.vllm.get_llm_outpt(input_embeds, vt_range).logits
                    before_edit_ids = torch.softmax(logits, -1).argmax(-1)[:, -label_ids.shape[1]:] # [1, l2]
                    rdl['predict_before_edit'] = tokenizer.decode(label_ids[label_masks.to(bool)])
                    edl['before_edit_ids'] = before_edit_ids
            # edit 
            start_t = time()
            editor.edit_one_piece(ed['requests'][0])
            rd['reliability'][0]['edit_time'] = time() - start_t
            # compute scores 
            rd = self.__get_results_after_edit__(editor.vllm, ed, rd)
            results.append(rd)
            # Restore to original model
            editor.restore_to_original_model()
        save_dir = os.path.join(self.result_dir, 'single_edit')
        # save results
        self.save_results(os.path.join(save_dir, 'results.json'), results)
        mean_results = self.get_mean_results(results)
        mean_results['sample_count'] = len(results)
        self.save_results(os.path.join(save_dir, 'mean_results.json'), mean_results)
        return results

    def evaluate_sequential_edit(self, edit_n = 10, random = False, seed = None):
        editor = self.editor
        print('Evaluating reliability, generality and locality for %s on %s with sequential editing %s.'
              %(*editor.name_of_editor_and_model(), edit_n))
        # preprocess data for sequential editing evaluation
        def split_data(data): 
            splited_data = []
            splited_data_ns = []
            now_split = []
            now_split_edit_n = 0
            for d in data:
                now_split.append(d)
                now_split_edit_n += len(d['requests'])
                if now_split_edit_n >= edit_n:
                    splited_data.append(now_split)
                    splited_data_ns.append(now_split_edit_n)
                    now_split = []
                    now_split_edit_n = 0
            return splited_data, splited_data_ns
        eval_data = deepcopy(self.eval_data.data_with_img)
        result_data = deepcopy(self.eval_data.data_with_img_path)
        if random:
            seed = seed if seed != None else np.random.randint(1, 999999)
            np.random.default_rng(seed).shuffle(eval_data)
            np.random.default_rng(seed).shuffle(result_data)
        eval_data, eval_data_ns = split_data(eval_data)
        result_data, _ = split_data(result_data)
        # evaluate
        tokenizer = editor.vllm.get_llm_tokenizer()
        editor.restore_to_original_model()
        results = [] 
        for split_rd, split_ed in zip(tqdm(result_data, 'Evaluating'), eval_data):
            split_res = []
            for rd, ed in zip(tqdm(split_rd, 'Preparing', leave = False), split_ed):
                rd['reliability'] = rd.pop('requests') 
                for r in rd['reliability']:
                    r['target'] = r.pop('target_new') 
                for loc_name in ed['locality'].keys(): # predict before edit for locality data
                    for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                        (input_embeds, vt_range), label_ids, label_masks = editor.vllm.prompts_imgs_target_to_xym(
                            [edl['prompt']], [edl['image']], [edl['target']])
                        logits = editor.vllm.get_llm_outpt(input_embeds, vt_range).logits
                        before_edit_ids = torch.softmax(logits, -1).argmax(-1)[:, -label_ids.shape[1]:] # [1, l2]
                        rdl['predict_before_edit'] = tokenizer.decode(before_edit_ids[label_masks.to(bool)])
                        edl['before_edit_ids'] = before_edit_ids
            for rd, ed in zip(tqdm(split_rd, 'Editing', leave = False), split_ed): # edit 
                for rdr, edr in zip(rd['reliability'], ed['requests']):
                    start_t = time()
                    editor.edit_one_piece(edr)
                    rdr['edit_time'] = time() - start_t
            for rd, ed in zip(tqdm(split_rd, 'Testing', leave = False), split_ed): # compute scores 
                rd = self.__get_results_after_edit__(editor.vllm, ed, rd)
                split_res.append(rd)
            editor.restore_to_original_model()
            results.append(split_res)
        # save results
        save_dir = os.path.join(self.result_dir, 'sequential_edit_%s'%edit_n)
        self.save_results(os.path.join(save_dir, '%sresults.json'%('seed_%s_'%seed if random else '')), results)
        split_mean = [self.get_mean_results(sr) for sr in results]
        for mr, n in zip(split_mean, eval_data_ns):
            mr['sequential_edit_n'] = n
        total_mean = self.get_mean_results([r for sr in results for r in sr])
        total_mean['total_edit_n'] = sum(eval_data_ns)
        mean_results = {"total_mean": total_mean, "split_mean": split_mean}
        self.save_results(os.path.join(save_dir, '%smean_results.json'%('seed_%s_'%seed if random else '')), mean_results)
        return results

    def __generate_free_running__(self, vllm:BaseVLLMForEdit, prompt, image, max_new_tokens):
        '''Workstream E0: greedy free-running generation through `get_llm_outpt` so the
        LiveEdit edit hooks keep firing at every decode step (HF `.generate()` would
        bypass them). Unlike the teacher-forced scorer this lets generation errors
        compound, which is exactly the regime where AR re-grounding should help.

        `query_range` is pinned to the image+prompt span so MoE retrieval always queries
        the prompt (not the growing generated suffix). For AR mode the routing-gate chunk
        index advances with the number of generated tokens.

        Returns (decoded_text, generated_token_ids).
        '''
        tokenizer = vllm.get_llm_tokenizer()
        editor = getattr(self, 'editor', None)
        chunk_size = max(int(getattr(self, 'ar_chunk_size', 16)), 1)
        ar_on = (editor is not None and getattr(editor, 'ar_mode', False)
                 and getattr(editor, 'ar_use_routing_gate', True))
        max_chunks = getattr(editor, 'max_chunks', None) if editor is not None else None
        eos_id = tokenizer.eos_token_id
        llm_inpt, vt_range = vllm.get_llm_input_embeds([prompt], [image])
        # image+prompt end = current sequence length before any generated token
        query_end = llm_inpt['inputs_embeds'].shape[1]
        gen_ids = []
        try:
            for _ in range(int(max_new_tokens)):
                llm_inpt['query_range'] = (0, query_end)
                if ar_on:
                    ci = len(gen_ids) // chunk_size
                    if max_chunks is not None:
                        ci = min(ci, max_chunks - 1)
                    editor.current_infer_chunk_index = ci
                with torch.no_grad():
                    logits = vllm.get_llm_outpt(llm_inpt, vt_range).logits
                next_id = int(logits[0, -1].argmax().item())
                if eos_id is not None and next_id == eos_id:
                    break
                gen_ids.append(next_id)
                emb = vllm.get_llm_embed_tokens(torch.tensor([[next_id]], dtype=torch.long))
                emb = emb.to(dtype=llm_inpt['inputs_embeds'].dtype, device=llm_inpt['inputs_embeds'].device)
                llm_inpt['inputs_embeds'] = torch.cat([llm_inpt['inputs_embeds'], emb], dim=1)
                am = llm_inpt['attention_mask']
                llm_inpt['attention_mask'] = torch.cat(
                    [am, torch.ones(1, 1, dtype=am.dtype, device=am.device)], dim=1)
        finally:
            if editor is not None and getattr(editor, 'ar_mode', False):
                editor.current_infer_chunk_index = None
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text, gen_ids

    def evaluate_sequential_edit_freegen(self, edit_n=10, random=False, seed=None,
                                         gen_eval_n=200, max_new_tokens=128):
        '''Workstream E0: free-running (non-teacher-forced) reliability evaluation.

        Applies edits exactly like `evaluate_sequential_edit` (same shuffle/seed/splits),
        but for reliability it *generates* the answer and stores the decoded text + target
        so sequence-level metrics (EM here; ROUGE-L/BERTScore/LLM-judge offline in E1/E2)
        can be computed. Runs on the first `gen_eval_n` reliability samples to bound the
        (much higher) generation cost. Generality/locality are skipped here.
        '''
        editor = self.editor
        print('Free-running reliability eval for %s on %s (edit_n=%s, gen_eval_n=%s, max_new_tokens=%s).'
              % (*editor.name_of_editor_and_model(), edit_n, gen_eval_n, max_new_tokens))

        def split_data(data):
            splited_data, splited_data_ns, now_split, now_n = [], [], [], 0
            for d in data:
                now_split.append(d)
                now_n += len(d['requests'])
                if now_n >= edit_n:
                    splited_data.append(now_split)
                    splited_data_ns.append(now_n)
                    now_split, now_n = [], 0
            return splited_data, splited_data_ns

        eval_data = deepcopy(self.eval_data.data_with_img)
        result_data = deepcopy(self.eval_data.data_with_img_path)
        if random:
            seed = seed if seed is not None else np.random.randint(1, 999999)
            np.random.default_rng(seed).shuffle(eval_data)
            np.random.default_rng(seed).shuffle(result_data)
        eval_data, _ = split_data(eval_data)
        result_data, _ = split_data(result_data)

        def _norm(s):
            return ' '.join(str(s).lower().split())

        editor.restore_to_original_model()
        results = []
        collected = 0
        for split_rd, split_ed in zip(tqdm(result_data, 'FreeGen'), eval_data):
            for rd in split_rd:
                rd['reliability'] = rd.pop('requests')
                for r in rd['reliability']:
                    r['target'] = r.pop('target_new')
            for rd, ed in zip(split_rd, split_ed):  # apply sequential edits for this split
                for edr in ed['requests']:
                    editor.edit_one_piece(edr)
            split_res = []
            for rd, ed in zip(split_rd, split_ed):  # free-running generation for reliability
                for rdr, edr in zip(rd['reliability'], ed['requests']):
                    target = edr.get('target_new', rdr.get('target'))
                    gen_text, gen_ids = self.__generate_free_running__(
                        editor.vllm, edr['prompt'], edr['image'], max_new_tokens)
                    rdr['prediction_freegen'] = gen_text
                    rdr['target_text'] = target
                    rdr['gen_token_count'] = len(gen_ids)
                    rdr['em_freegen'] = float(_norm(gen_text) == _norm(target))
                    collected += 1
                split_res.append(rd)
            editor.restore_to_original_model()
            results.append(split_res)
            if collected >= gen_eval_n:
                break

        save_dir = os.path.join(self.result_dir, 'sequential_edit_%s' % edit_n)
        prefix = 'seed_%s_' % seed if random else ''
        self.save_results(os.path.join(save_dir, '%sfreegen_results.json' % prefix), results)
        ems = [rr['em_freegen'] for sr in results for rd in sr for rr in rd['reliability']]
        gtc = [rr['gen_token_count'] for sr in results for rd in sr for rr in rd['reliability']]
        mean_results = {"freegen": {
            "em": float(np.mean(ems)) if ems else None,
            "count": len(ems),
            "mean_gen_token_count": float(np.mean(gtc)) if gtc else None,
            "edit_n": edit_n,
            "max_new_tokens": int(max_new_tokens),
        }}
        self.save_results(os.path.join(save_dir, '%sfreegen_mean_results.json' % prefix), mean_results)
        print('Free-running EM=%.4f over %d samples.' % (mean_results['freegen']['em'] or 0.0, len(ems)))
        return results

    def __get_results_after_edit__(self, vllm:BaseVLLMForEdit, ed, rd):
        def get_eval_xym(prompt, image, target):
            (x, vt_range), y, m = vllm.prompts_imgs_target_to_xym([prompt], [image], [target])
            x['query_triple'] = (prompt, image, target)
            x['query_range'] = (0, x['inputs_embeds'].shape[1] - m.shape[1] + 1)
            return (x, vt_range), y, m
        def accuracy_and_prediction(input_embeds, vt_range, label_ids, label_masks):
            # label_ids/label_masks: [1, l2]
            assert len(label_ids) == 1 and len(label_masks) == 1
            l2 = label_ids.shape[1]
            logits = vllm.get_llm_outpt(input_embeds, vt_range).logits # [1,l1,d]
            pre_y = torch.softmax(logits, -1).argmax(-1) # [1, l1]
            pre_y = pre_y[:, -l2:]  # take last l2 positions
            # align length: if model output was shorter than l2, pad on the left so indexing is safe
            if pre_y.shape[1] < l2:
                pad_len = l2 - pre_y.shape[1]
                pre_y = torch.cat([
                    torch.zeros(1, pad_len, dtype=pre_y.dtype, device=pre_y.device),
                    pre_y
                ], dim=1)
            denom = label_masks.sum()
            acc = ((pre_y == label_ids) * label_masks).sum() / denom if denom > 0 else 0.0
            return float(acc), pre_y
        tokenizer = vllm.get_llm_tokenizer()
        chunk_size = getattr(self, 'ar_chunk_size', 16)
        editor = getattr(self, 'editor', None)
        # reliability
        for rdr, edr in zip(rd['reliability'], ed['requests']):
            (input_embeds, vt_range), label_ids, label_masks = get_eval_xym(
                    edr['prompt'], edr['image'], edr['target_new'])
            # AR routing gate: set current_infer_chunk_index to last chunk of this request when gate on
            if editor is not None and getattr(editor, 'ar_mode', False) and getattr(editor, 'ar_use_routing_gate', True):
                from editor.vllm_editors.liveedit.chunk_utils import split_target_into_chunks
                chunks = split_target_into_chunks(
                    tokenizer, edr['target_new'],
                    getattr(editor, 'chunk_size', 16),
                    getattr(editor, 'max_chunks', None))
                if chunks:
                    editor.current_infer_chunk_index = len(chunks) - 1
            try:
                acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, label_ids, label_masks)
            finally:
                if editor is not None and getattr(editor, 'ar_mode', False):
                    editor.current_infer_chunk_index = None
            # Move to CPU for indexing/compare; align to label length so boolean indexing is in bounds
            pre_y = pre_y.to(torch.device('cpu'))
            label_ids_c = label_ids.to(torch.device('cpu'))
            label_masks_c = label_masks.to(torch.device('cpu'))
            l2 = label_masks_c.shape[1]
            if pre_y.shape[1] < l2:
                pre_y = torch.cat([torch.zeros(1, l2 - pre_y.shape[1], dtype=pre_y.dtype), pre_y], dim=1)
            elif pre_y.shape[1] > l2:
                pre_y = pre_y[:, -l2:]
            rdr['predict_after_edit'] = tokenizer.decode(pre_y[label_masks_c.to(bool)])
            rdr['acc'] = acc
            # AR-specific: EM, target token count, chunk EM/F1
            n_tok = label_masks_c.sum().item()
            rdr['target_token_count'] = n_tok
            if n_tok > 0:
                pm = pre_y[label_masks_c.to(bool)]
                lm = label_ids_c[label_masks_c.to(bool)]
                em = (pm == lm).all().item() if pm.numel() == lm.numel() else False
                rdr['em'] = float(em)
                chunk_em, chunk_f1 = _chunk_em_f1(pre_y, label_ids_c, label_masks_c, chunk_size)
                rdr['chunk_em'] = chunk_em
                rdr['chunk_f1'] = chunk_f1
                # Workstream A: per-position correctness over masked target tokens (in order)
                # enables positional efficacy decay analysis (accuracy vs position in long answer)
                rdr['pos_hits'] = (pm == lm).to(torch.int).tolist() if pm.numel() == lm.numel() else []
            else:
                rdr['em'] = 1.0
                rdr['chunk_em'] = 0.0
                rdr['chunk_f1'] = 0.0
                rdr['pos_hits'] = []
        # generality
        for gen_name in ed['generality']:
            for rdg, edg in zip(rd['generality'][gen_name], ed['generality'][gen_name]):
                (input_embeds, vt_range), label_ids, label_masks = get_eval_xym(
                    edg['prompt'], edg['image'], edg['target'])
                acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, label_ids, label_masks)
                pre_y_c = pre_y.cpu()
                label_masks_c = label_masks.cpu()
                if pre_y_c.shape[1] != label_masks_c.shape[1]:
                    pre_y_c = pre_y_c[:, -label_masks_c.shape[1]:] if pre_y_c.shape[1] >= label_masks_c.shape[1] else torch.cat([torch.zeros(1, label_masks_c.shape[1] - pre_y_c.shape[1], dtype=pre_y_c.dtype), pre_y_c], dim=1)
                rdg['predict_after_edit'] = tokenizer.decode(pre_y_c[label_masks_c.to(bool)])
                rdg['acc'] = acc
        # locality
        for loc_name in ed['locality']:
            for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                (input_embeds, vt_range), _, label_masks = get_eval_xym(
                    edl['prompt'], edl['image'], edl['target'])
                acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, edl['before_edit_ids'], label_masks)
                pre_y_c = pre_y.cpu()
                label_masks_c = label_masks.cpu()
                if pre_y_c.shape[1] != label_masks_c.shape[1]:
                    pre_y_c = pre_y_c[:, -label_masks_c.shape[1]:] if pre_y_c.shape[1] >= label_masks_c.shape[1] else torch.cat([torch.zeros(1, label_masks_c.shape[1] - pre_y_c.shape[1], dtype=pre_y_c.dtype), pre_y_c], dim=1)
                rdl['predict_after_edit'] = tokenizer.decode(pre_y_c[label_masks_c.to(bool)])
                rdl['acc'] = acc
        return rd

    def get_mean_results(self, results:List[Dict]):
        """Get numbers from a result: {
            "reliability": [
                {"acc": float, "edit_time": float}, 
                {"acc": float, "edit_time": float}, ...]
            "generality": {
                sub_metric_1: [{"acc": float}, {"acc": float}, ...], 
                sub_metric_2: [{"acc": float}, {"acc": float}, ...], ...}
            "locality": {
                sub_metric_1: [{"acc": float}, {"acc": float}, ...], 
                sub_metric_2: [{"acc": float}, {"acc": float}, ...], ...}
        }
        """
        mean_res = {"reliability": {}, "generality": {}, "locality": {}}
        # sum values
        for r in results:
            for rr in r['reliability']:
                for value_name, value in rr.items():
                    if isinstance(value, (int, float)):
                        if value_name not in mean_res['reliability']:
                            mean_res['reliability'][value_name] = [0, 0]
                        mean_res['reliability'][value_name][0] += value
                        mean_res['reliability'][value_name][1] += 1
            for sub_metric in r['generality'].keys():
                if sub_metric not in mean_res['generality']:
                    mean_res['generality'][sub_metric] = {}
                for sub_res in r['generality'][sub_metric]:
                    for value_name, value in sub_res.items():
                        if isinstance(value, (int, float)):
                            if value_name not in mean_res['generality'][sub_metric]:
                                mean_res['generality'][sub_metric][value_name] = [0, 0]
                            mean_res['generality'][sub_metric][value_name][0] += value
                            mean_res['generality'][sub_metric][value_name][1] += 1
            for sub_metric in r['locality'].keys():
                if sub_metric not in mean_res['locality']:
                    mean_res['locality'][sub_metric] = {}
                for sub_res in r['locality'][sub_metric]:
                    for value_name, value in sub_res.items():
                        if isinstance(value, (int, float)):
                            if value_name not in mean_res['locality'][sub_metric]:
                                mean_res['locality'][sub_metric][value_name] = [0, 0]
                            mean_res['locality'][sub_metric][value_name][0] += value
                            mean_res['locality'][sub_metric][value_name][1] += 1
        # compute mean results
        for value_name, value in mean_res['reliability'].items():
            mean_res['reliability'][value_name] = value[0] / value[1]
        for sub_metric in mean_res['generality'].keys():
            for value_name, value in mean_res['generality'][sub_metric].items():
                mean_res['generality'][sub_metric][value_name] = value[0] / value[1]
        for sub_metric in mean_res['locality'].keys():
            for value_name, value in mean_res['locality'][sub_metric].items():
                mean_res['locality'][sub_metric][value_name] = value[0] / value[1]
        # AR-specific: length-bin EM/F1 and chunk EM/F1 (bins: <=16, 17-32, 33-64, 65+)
        bin_names = ["<=16", "17-32", "33-64", "65+"]
        bin_agg = {b: {"em": [], "acc": [], "chunk_em": [], "chunk_f1": []} for b in bin_names}
        for r in results:
            for rr in r.get("reliability", []):
                n = rr.get("target_token_count")
                if n is None:
                    continue
                b = _length_bin_name(int(n))
                if "em" in rr:
                    bin_agg[b]["em"].append(rr["em"])
                if "acc" in rr:
                    bin_agg[b]["acc"].append(rr["acc"])
                if "chunk_em" in rr:
                    bin_agg[b]["chunk_em"].append(rr["chunk_em"])
                if "chunk_f1" in rr:
                    bin_agg[b]["chunk_f1"].append(rr["chunk_f1"])
        mean_res["length_bin"] = {}
        for b in bin_names:
            arr = bin_agg[b]
            count = len(arr["em"]) if arr["em"] else 0
            mean_res["length_bin"][b] = {
                "count": count,
                "em": float(np.mean(arr["em"])) if arr["em"] else None,
                "acc": float(np.mean(arr["acc"])) if arr["acc"] else None,
                "chunk_em": float(np.mean(arr["chunk_em"])) if arr["chunk_em"] else None,
                "chunk_f1": float(np.mean(arr["chunk_f1"])) if arr["chunk_f1"] else None,
            }
        # Workstream A: positional efficacy decay curve.
        # For each reliability sample, map each masked target token to a normalized
        # position decile (i / len), then average correctness per decile across samples.
        n_pos_bins = 10
        pos_sum = [0.0] * n_pos_bins
        pos_cnt = [0] * n_pos_bins
        for r in results:
            for rr in r.get("reliability", []):
                hits = rr.get("pos_hits")
                if not hits:
                    continue
                Lh = len(hits)
                for i, h in enumerate(hits):
                    b = min(int(n_pos_bins * i / Lh), n_pos_bins - 1)
                    pos_sum[b] += float(h)
                    pos_cnt[b] += 1
        mean_res["position_curve"] = {
            "n_bins": n_pos_bins,
            "scale": "normalized",  # decile of position within target (0=start, 9=end)
            "acc": [(pos_sum[i] / pos_cnt[i]) if pos_cnt[i] else None for i in range(n_pos_bins)],
            "count": pos_cnt,
        }
        return mean_res

    def save_results(self, save_path:str, results:Dict, decimal = 4):
        def set_decimal(r):
            if isinstance(r, list):
                for i in range(len(r)):
                    r[i] = set_decimal(r[i])
            elif isinstance(r, dict) or isinstance(r, defaultdict):
                for k in r.keys():
                    r[k] = set_decimal(r[k])
            elif isinstance(r, float):
                r = round(r, decimal)
            return r
        res = deepcopy(results)
        res = set_decimal(res)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(os.path.join(save_path), 'w') as f:
            json.dump(res, f, indent = 4)


