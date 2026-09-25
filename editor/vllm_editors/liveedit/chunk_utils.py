"""
Chunk utility for AR-LiveEdit: split target_new into token chunks and reconstruct text.
Isolated from training/inference; used for Task 2 validation.
"""
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def split_target_into_chunks(
    tokenizer: "PreTrainedTokenizerBase",
    target_text: str,
    chunk_size: int,
    max_chunks: Union[int, None] = None,
) -> List[List[int]]:
    """
    Split target text into token id chunks of at most chunk_size tokens each.

    Args:
        tokenizer: LLM tokenizer (e.g. from editor.vllm.get_llm_tokenizer()).
        target_text: The target string to chunk (e.g. target_new).
        chunk_size: Maximum tokens per chunk.
        max_chunks: If set, return at most this many chunks (last chunk may be partial).
            None means no limit.

    Returns:
        List of chunk token id lists. Empty target yields [].
    """
    if not target_text.strip():
        return []
    enc = tokenizer(
        target_text,
        return_tensors=None,
        add_special_tokens=False,
    )
    # Handle BatchEncoding (dict with input_ids), Encoding (has .ids), or list
    if hasattr(enc, "input_ids"):
        token_ids = enc.input_ids
    elif hasattr(enc, "ids"):
        token_ids = enc.ids
    elif isinstance(enc, dict):
        token_ids = enc["input_ids"]
    else:
        token_ids = enc
    if not token_ids:
        return []
    # Single sequence: list of ints; batch of one: list of one list
    ids = list(token_ids) if isinstance(token_ids[0], int) else list(token_ids[0])

    chunks: List[List[int]] = []
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        chunks.append(chunk)
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    return chunks


def reconstruct_text_from_chunks(
    tokenizer: "PreTrainedTokenizerBase",
    chunks: List[List[int]],
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = False,
) -> str:
    """
    Reconstruct full target text from token id chunks.

    Args:
        tokenizer: Same tokenizer used for split_target_into_chunks.
        chunks: List of token id lists from split_target_into_chunks.
        skip_special_tokens: Passed to tokenizer.decode.
        clean_up_tokenization_spaces: Passed to tokenizer.decode (False helps round-trip).

    Returns:
        Decoded string. Empty chunks yields "".
    """
    if not chunks:
        return ""
    flat = []
    for c in chunks:
        flat.extend(c)
    return tokenizer.decode(
        flat,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )


def positional_chunk_index(
    query_token_offset: int,
    chunk_size: int,
    max_chunks: Union[int, None] = None,
) -> int:
    """Token-position AR chunk index (baseline LiveEdit)."""
    chunk_size = max(int(chunk_size), 1)
    idx = int(query_token_offset) // chunk_size
    if max_chunks is not None:
        idx = min(idx, max(int(max_chunks), 1) - 1)
    return max(idx, 0)


def vision_gated_chunk_index(
    cross_attn_qp,
    query_token_offset: int,
    chunk_size: int,
    max_chunks: Union[int, None] = None,
    n_qformer_queries: int = 32,
) -> int:
    """Step 5 spike 4c: pick AR chunk from Q-Former image cross-attention peaks.

    Maps the dominant Q-Former query slot (highest patch mass) to a chunk bin,
    falling back to positional chunking when attention is unavailable.
    """
    token_chunk = positional_chunk_index(query_token_offset, chunk_size, max_chunks)
    if cross_attn_qp is None:
        return token_chunk
    if not hasattr(cross_attn_qp, "shape"):
        return token_chunk
    attn = cross_attn_qp.float()
    if attn.dim() == 3:
        attn = attn[0]
    if attn.numel() == 0:
        return token_chunk
    query_mass = attn.sum(dim=1)
    peak_q = int(query_mass.argmax().item())
    n_chunks = max(int(max_chunks) if max_chunks is not None else 1, 1)
    vision_chunk = int(peak_q * n_chunks / max(int(n_qformer_queries), 1))
    vision_chunk = min(max(vision_chunk, 0), n_chunks - 1)
    return vision_chunk


def infer_ar_chunk_index(
    cross_attn_qp,
    query_token_offset: int,
    chunk_size: int,
    max_chunks: Union[int, None] = None,
    vision_gated: bool = False,
    n_qformer_queries: int = 32,
) -> int:
    """Unified AR chunk index: positional (legacy) or vision-gated (Step 5 4c)."""
    if vision_gated:
        return vision_gated_chunk_index(
            cross_attn_qp,
            query_token_offset,
            chunk_size,
            max_chunks,
            n_qformer_queries=n_qformer_queries,
        )
    return positional_chunk_index(query_token_offset, chunk_size, max_chunks)
