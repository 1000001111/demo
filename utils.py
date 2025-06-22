# -----------------------------------------------------------------------------
# --- utils.py ---
"""Misc helpers."""
import numpy as np

def bits_to_int(b):
    b = np.asarray(b)
    k = b.shape[-1]
    return b.dot(1 << np.arange(k)[::-1])

def int_to_bits(v, k):
    return ((v[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(int)