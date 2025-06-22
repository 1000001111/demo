# -----------------------------------------------------------------------------
# --- ofdm.py ---
"""Simple CP‑OFDM mod/demod for NR (30 kHz SCS, NFFT = 4096)."""
import numpy as np

NFFT_DEFAULT = 4096


def ofdm_mod(grid: np.ndarray, Nfft: int = NFFT_DEFAULT, cp_lens=None) -> np.ndarray:
    n_sym = grid.shape[1]
    if cp_lens is None:
        cp_lens = [288] + [256] * (n_sym - 1)
    out = []
    n_sc = grid.shape[0]
    start = (Nfft - n_sc) // 2
    for s in range(n_sym):
        X = np.zeros(Nfft, dtype=complex)
        X[start : start + n_sc] = grid[:, s]
        x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(Nfft)
        out.append(x[-cp_lens[s]:])
        out.append(x)
    return np.concatenate(out)


def ofdm_demod(sig: np.ndarray, *, n_sc: int, n_sym: int, Nfft: int = NFFT_DEFAULT, cp_lens=None) -> np.ndarray:
    if cp_lens is None:
        cp_lens = [288] + [256] * (n_sym - 1)
    grid = np.zeros((n_sc, n_sym), dtype=complex)
    pos = 0
    start = (Nfft - n_sc) // 2
    for s in range(n_sym):
        x = sig[pos + cp_lens[s] : pos + cp_lens[s] + Nfft]
        pos += cp_lens[s] + Nfft
        X = np.fft.fftshift(np.fft.fft(x) / np.sqrt(Nfft))
        grid[:, s] = X[start : start + n_sc]
    return grid
