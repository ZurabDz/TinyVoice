import numpy as np


def collapse_ctc_ids(token_ids, blank_id: int = 0) -> list[int]:
    """Collapse repeats and drop blanks from a token stream."""
    collapsed = []
    previous = blank_id
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id != previous and token_id != blank_id:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


def greedy_ctc_decode(logits: np.ndarray, length: int, blank_id: int = 0) -> list[int]:
    """Greedy CTC decode: argmax then collapse repeats and blanks."""
    token_ids = np.argmax(logits[:length], axis=-1)
    return collapse_ctc_ids(token_ids, blank_id=blank_id)


def decode_token_ids(token_ids, tokenizer) -> str:
    """Decode token ids while skipping the tokenizer's blank and pad ids."""
    ignored_ids = {int(tokenizer.blank_id)}
    pad_id = getattr(tokenizer, "label_pad_token", None)
    if pad_id is not None:
        ignored_ids.add(int(pad_id))
    return tokenizer.decode(int(token_id) for token_id in token_ids if int(token_id) not in ignored_ids)


def greedy_ctc_decode_text(logits: np.ndarray, length: int, tokenizer) -> str:
    """Greedy CTC decode followed by tokenizer decoding."""
    token_ids = greedy_ctc_decode(logits, length, blank_id=int(tokenizer.blank_id))
    return decode_token_ids(token_ids, tokenizer)
