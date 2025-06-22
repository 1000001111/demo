# -----------------------------------------------------------------------------
# --- ce_eq.py ---
"""LS/MMSE channel estimation + ZF/MMSE equalization."""
import numpy as np


def ls_ce(y_dmrs: np.ndarray, x_dmrs: np.ndarray):
    return y_dmrs / x_dmrs


def mmse_ce(y_dmrs, x_dmrs, noise_var, ch_var=1.0):
    ls = ls_ce(y_dmrs, x_dmrs)
    w = ch_var / (ch_var + noise_var)
    return w * ls


def interp_freq(h_est_pilots: np.ndarray, pilot_idx: np.ndarray, n_sc: int):
    idx_full = np.arange(n_sc)
    return np.interp(idx_full, pilot_idx, h_est_pilots)


def zf_eq(y: np.ndarray, h_hat: np.ndarray, eps=1e-9):
    return y / (h_hat + eps)


def mmse_eq(y, h_hat, noise_var):
    return h_hat.conj() * y / (np.abs(h_hat) ** 2 + noise_var)
