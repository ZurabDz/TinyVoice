import jax.numpy as jnp


def edit_distance(a, b):
    """Levenshtein distance between two integer sequences."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def to_jax(elem):
    """Transfer a batch tuple to JAX device arrays."""
    return (
        jnp.array(elem[0], dtype=jnp.float32),  # audios
        jnp.array(elem[1], dtype=jnp.int32),  # frames
        jnp.array(elem[2], dtype=jnp.int32),  # labels
        jnp.array(elem[3], dtype=jnp.int32),  # label_lengths
    )
