# pusch_sim.py — Mini-project: PUSCH channel estimation/equalisation – Requirement I
# Author: ChatGPT (generated 2025-06-21)
"""
Pipeline implemented
--------------------
1. Read DMRS from Excel → column 3 of the resource-grid (14 OFDM symbols, 3rd symbol index 3).
2. Generate random bits (3276 × 13 × 4) and map to 16-QAM (Gray, 3GPP TS 38.211).
3. Populate the resource-grid: data → all columns except 3, scan down each column.
4. Insert DMRS into column 3.
5. OFDM modulation (IFFT 4096) + variable CP (320 for sym 0,7; 288 otherwise) → 61 440 samples.
6. Pass through TDL-B 100 ns (25 km/h) fading channel + AWGN @ 20 dB SNR.
7. OFDM demod & 16-QAM hard-decision.
8. Compute MSE (complex samples) & BER.

Python ≥3.9, numpy, pandas, scipy required.
"""

import numpy as np
import pandas as pd
from scipy import signal

# ---------- Parameters ----------
N_SC = 3276           # used subcarriers per OFDM symbol
N_SYM = 14            # OFDM symbols/slot
FFT_SIZE = 4096       # IFFT/FFT length (15 kHz SCS → Fs = 61.44 MHz)
CP_SHORT = 288        # samples
CP_LONG  = 320        # samples (symbols 0 & 7)
DATA_COLS = [c for c in range(N_SYM) if c != 3]  # 13 data columns
SNR_dB = 20

# ---------- Helpers ----------

def gray_16qam_mod(bits: np.ndarray) -> np.ndarray:
    """Map bits (B,4) → 16-QAM symbols with unit average power."""
    assert bits.shape[1] == 4, "Expect 4 bits per symbol"
    b0, b1, b2, b3 = bits.T
    I = 2 * (2 * b0 + b1) - 3  # {-3,-1, +1, +3}
    Q = 2 * (2 * b2 + b3) - 3
    symbols = (I + 1j * Q) / np.sqrt(10)
    return symbols


def gray_16qam_demod_hard(sym: np.ndarray) -> np.ndarray:
    """Hard-decision demapper for 16-QAM, returns bits shape (N,4)."""
    # Denormalise
    sym *= np.sqrt(10)
    I = sym.real
    Q = sym.imag

    b0 = (I < 0).astype(int)
    b1 = ((np.abs(I) < 2)).astype(int) ^ b0  # Gray decode
    b2 = (Q < 0).astype(int)
    b3 = ((np.abs(Q) < 2)).astype(int) ^ b2
    return np.stack([b0, b1, b2, b3], axis=1)


def parse_dmrs_cell(val: str):
    s = str(val).strip()
    if s == "0":
        return 0j
    return complex(s.replace("i", "j"))


def build_tdlb_100ns_fs61440() -> np.ndarray:
    """Construct discrete TDL-B(100 ns, 25 km/h) impulse response at Fs=61.44 MHz."""
    # Power (dB) & delays (ns) from 3GPP 38.901 Table 7.7.1-1 (scaled 100 ns rms)
    # Only first 23 taps (others negligible)
    delays_ns = np.array([0, 30, 70, 90, 110, 190, 410, 560, 710, 870,
                          1090, 1730, 2510, 3110, 3750, 4300, 4900,
                          5800, 6100, 7100, 8100, 8700, 9100])
    power_dB   = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, -2.0, -3.0, -8.0, -17.2,
                           -20.8, -21.0, -23.0, -24.8, -27.7, -30.8, -31.5, -33.5,
                           -35.5, -36.7, -40.0, -41.0, -41.3, -42.3])
    Fs = 61.44e6             # 15 kHz SCS
    Ts = 1 / Fs
    delays_samp = np.round(delays_ns * 1e-9 / Ts).astype(int)
    L = delays_samp.max() + 1
    h = np.zeros(L, dtype=complex)
    for d, p in zip(delays_samp, power_dB):
        # Rayleigh fading tap
        coeff = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        h[d] += coeff * 10 ** (p / 20)
    # Normalise to unit energy
    h /= np.sqrt(np.sum(np.abs(h) ** 2))
    return h


def awgn(sig: np.ndarray, snr_dB: float) -> np.ndarray:
    sig_power = np.mean(np.abs(sig) ** 2)
    snr_linear = 10 ** (snr_dB / 10)
    noise_power = sig_power / snr_linear
    noise = (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)) * np.sqrt(noise_power / 2)
    return sig + noise


# ---------- 1. Read DMRS ----------
DMRS_FILE = "l1_mini_project_ce_eq_dmrs_data.xlsx"
dmrs_raw = pd.read_excel(DMRS_FILE, header=None)[0].apply(parse_dmrs_cell).values.astype(complex)
assert dmrs_raw.shape[0] == N_SC, "DMRS length mismatch"

# ---------- 2. Generate random bits & 16-QAM map ----------
rand_bits = np.random.randint(0, 2, size=(N_SC * len(DATA_COLS), 4), dtype=int)
qam_syms = gray_16qam_mod(rand_bits)

# ---------- 3. Populate resource grid ----------
resource_grid = np.zeros((N_SC, N_SYM), dtype=complex)
resource_grid[:, 3] = dmrs_raw  # insert DMRS

# Map data column-wise (top→bottom, left→right – excluding col 3)
idx = 0
for col in DATA_COLS:
    resource_grid[:, col] = qam_syms[idx: idx + N_SC]
    idx += N_SC
assert idx == qam_syms.size, "Data mapping error"

# ---------- 4. OFDM modulation ----------
cp_lengths = [CP_LONG if s in (0, 7) else CP_SHORT for s in range(N_SYM)]
assert sum(cp_lengths) + FFT_SIZE * N_SYM == 61_440, "Sample count != 61 440"

offset = (FFT_SIZE - N_SC) // 2  # centre allocation
ofdm_time = []
for sym_idx in range(N_SYM):
    freq_vec = np.zeros(FFT_SIZE, dtype=complex)
    freq_vec[offset: offset + N_SC] = resource_grid[:, sym_idx]
    # shift so DC at index 0 → ifftshift
    freq_vec_shifted = np.fft.ifftshift(freq_vec)
    time_sym = np.fft.ifft(freq_vec_shifted)
    # append CP
    cp = time_sym[-cp_lengths[sym_idx]:]
    ofdm_time.append(np.concatenate([cp, time_sym]))

tx_signal = np.concatenate(ofdm_time)
assert tx_signal.size == 61_440

# ---------- 5. Channel (TDL-B + AWGN) ----------
channel_impulse = build_tdlb_100ns_fs61440()
rx_signal = signal.lfilter(channel_impulse, [1.0], tx_signal)
rx_signal = awgn(rx_signal, SNR_dB)

# ---------- 6. OFDM demod ----------
rx_ptr = 0
rx_grid = np.zeros_like(resource_grid)
for sym_idx in range(N_SYM):
    cp_len = cp_lengths[sym_idx]
    sym_samples = rx_signal[rx_ptr + cp_len: rx_ptr + cp_len + FFT_SIZE]
    rx_ptr += cp_len + FFT_SIZE
    freq_vec = np.fft.fft(sym_samples)
    freq_vec = np.fft.fftshift(freq_vec)
    rx_grid[:, sym_idx] = freq_vec[offset: offset + N_SC]

# ---------- 7. Metrics ----------
# Extract data symbols for MSE & BER
rx_data_syms = np.hstack([rx_grid[:, c] for c in DATA_COLS])

tx_data_syms = qam_syms
MSE = np.mean(np.abs(tx_data_syms - rx_data_syms) ** 2)

# Demap bits
rx_bits = gray_16qam_demod_hard(rx_data_syms)
BER = np.mean(rx_bits.reshape(-1) != rand_bits.reshape(-1))

print(f"MSE (complex symbol): {MSE:.4e}")
print(f"BER: {BER:.4e}")
