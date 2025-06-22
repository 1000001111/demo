# -----------------------------------------------------------------------------
# --- channel.py ---
"""Simplified TDL‑B (100 ns) fading + AWGN."""
import numpy as np
from scipy.signal import lfilter

class TDLBChannel:
    def __init__(self, delay_spread_ns=100, fd=25, fs=61.44e6, n_taps=6):
        self.fs = fs
        self.fd = fd
        self.n_taps = n_taps
        delays = np.linspace(0, delay_spread_ns * 1e-9, n_taps)
        self.sample_delays = np.round(delays * fs).astype(int)
        powers_db = -3 * np.arange(n_taps)  # simple 3 dB decay
        self.powers = 10 ** (powers_db / 10)

    def _generate_impulse_response(self):
        h = (
            np.random.randn(self.n_taps) + 1j * np.random.randn(self.n_taps)
        ) / np.sqrt(2)
        h *= np.sqrt(self.powers)
        h_time = np.zeros(self.sample_delays.max() + 1, complex)
        h_time[self.sample_delays] = h
        return h_time

    def __call__(self, x: np.ndarray):
        h = self._generate_impulse_response()
        y = lfilter(h, [1.0], x)
        return y, h

def awgn(x: np.ndarray, snr_db: float):
    sig_pow = np.mean(np.abs(x) ** 2)
    snr_lin = 10 ** (snr_db / 10)
    noise_pow = sig_pow / snr_lin
    n = np.sqrt(noise_pow / 2) * (
        np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
    )
    return x + n, noise_pow