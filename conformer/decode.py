import numpy as np


def greedy_ctc_decode(logits: np.ndarray, length: int, blank_id: int = 0) -> list[int]:
    """Greedy CTC decoding: argmax collapse removing blanks and repeats."""
    ids = np.argmax(logits[:length], axis=-1)
    result, prev = [], blank_id
    for token_id in ids:
        if token_id != prev and token_id != blank_id:
            result.append(int(token_id))
        prev = token_id
    return result


def beam_ctc_decode(
    logits: np.ndarray, length: int, blank_id: int = 0, beam_size: int = 10
) -> list[int]:
    """Prefix beam search CTC decoding (no language model).

    Args:
        logits: Raw model output of shape (T, V).
        length: Number of valid time steps.
        blank_id: Index of the CTC blank token.
        beam_size: Number of beams to keep at each step.

    Returns:
        List of token ids for the best hypothesis.
    """
    # Stable log-softmax
    l = logits[:length]
    l = l - l.max(axis=-1, keepdims=True)
    log_probs = l - np.log(np.exp(l).sum(axis=-1, keepdims=True))

    # beams: prefix_tuple -> (log_prob_ending_in_blank, log_prob_ending_in_non_blank)
    NEG_INF = -np.inf
    beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG_INF)}

    for t in range(length):
        lp = log_probs[t]
        new_beams: dict[tuple, tuple[float, float]] = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_total = np.logaddexp(p_b, p_nb)

            # Extend with blank — prefix unchanged
            cur = new_beams.get(prefix, (NEG_INF, NEG_INF))
            new_beams[prefix] = (np.logaddexp(cur[0], p_total + lp[blank_id]), cur[1])

            # Extend with each non-blank token
            for c in range(lp.shape[0]):
                if c == blank_id:
                    continue
                new_prefix = prefix + (c,)
                if prefix and prefix[-1] == c:
                    # Repeated last token: only blank-ending paths can extend
                    add_p = p_b + lp[c]
                else:
                    add_p = p_total + lp[c]
                cur = new_beams.get(new_prefix, (NEG_INF, NEG_INF))
                new_beams[new_prefix] = (cur[0], np.logaddexp(cur[1], add_p))

        # Prune to beam_size
        beams = dict(
            sorted(new_beams.items(), key=lambda x: np.logaddexp(*x[1]), reverse=True)[
                :beam_size
            ]
        )

    best = max(beams, key=lambda p: np.logaddexp(*beams[p]))
    return list(best)
