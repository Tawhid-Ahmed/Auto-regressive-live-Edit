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
