# --- qam.py ---
"""QAM modulation/demodulation utilities for mini‑project."""
import numpy as np

# Gray mapping tables for 4, 16, 64‑QAM (normalized to average power = 1)
_QAM_MAP = {
    4: np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2),
    16: None,
    64: None,
}

def _gen_16qam():
    # Gray-coded 16QAM, I‑bits b1b0, Q‑bits b3b2 (TS 38.211 §5.1.3‑2)
    I = np.array([-3, -1, +3, +1])
    Q = np.array([+3, +1, -3, -1])
    const = (I[None, :] + 1j * Q[:, None]).reshape(-1) / np.sqrt(10)
    return const

def _gen_64qam():
    # Generate Gray‑coded 64QAM constellation (TS 38.211 §5.1.3‑3)
    points = [-7, -5, -1, -3, +7, +5, +1, +3]
    I = np.array(points)
    Q = np.array(points[::-1])  # Gray on Q
    const = (I[None, :] + 1j * Q[:, None]).reshape(-1) / np.sqrt(42)
    return const

_QAM_MAP[16] = _gen_16qam()
_QAM_MAP[64] = _gen_64qam()

def qam_constellation(M: int):
    """Return Gray‑coded normalized constellation for M‑QAM (M = 4,16,64)."""
    return _QAM_MAP[M]


def qam_mod(bits: np.ndarray, M: int = 16) -> np.ndarray:
    """Map bits (..., k) → complex symbols. bits dtype int{0,1}."""
    k = int(np.log2(M))
    b = bits.reshape(-1, k)
    ints = b.dot(1 << np.arange(k)[::-1])  # binary → int
    const = qam_constellation(M)
    return const[ints].reshape(bits.shape[:-1])


def qam_demod(sym: np.ndarray, M: int = 16) -> np.ndarray:
    """Hard demapper: complex symbols → bits (..., k)."""
    const = qam_constellation(M)
    d2 = np.abs(sym.reshape(-1, 1) - const[None, :]) ** 2
    idx = d2.argmin(axis=1)
    k = int(np.log2(M))
    bits = ((idx[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(np.int8)
    return bits.reshape(sym.shape + (k,))