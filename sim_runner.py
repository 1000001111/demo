# -----------------------------------------------------------------------------
# --- sim_runner.py ---
"""End‑to‑end PUSCH CE/EQ demo simulation."""
import numpy as np
from qam import qam_mod, qam_demod
from ofdm import ofdm_mod, ofdm_demod
from channel import TDLBChannel, awgn
from ce_eq import ls_ce, interp_freq, zf_eq

# NR slot parameters
N_SC = 3276  # 273 RB × 12 subcarriers
N_SYM = 14   # 1 OFDM slot
DMRS_COL = 3  # column index of DMRS (0‑based)
CP_LENS = [288] + [256] * (N_SYM - 1)


def build_grid(data_syms: np.ndarray, dmrs_syms: np.ndarray):
    grid = np.zeros((N_SC, N_SYM), complex)
    grid[:, DMRS_COL] = dmrs_syms
    data_cols = [c for c in range(N_SYM) if c != DMRS_COL]
    grid[:, data_cols] = data_syms
    return grid


def run_sim(snr_db_list=np.arange(0, 32, 2), M=16, n_frames=100):
    k = int(np.log2(M))
    dmrs_seq = np.exp(1j * 2 * np.pi * np.random.rand(N_SC))  # pseudo‑random DMRS
    ch = TDLBChannel()
    ber = np.zeros_like(snr_db_list, float)
    for i_snr, snr_db in enumerate(snr_db_list):
        err = tot = 0
        for _ in range(n_frames):
            bits = np.random.randint(0, 2, size=(N_SC, N_SYM - 1, k))
            data_syms = qam_mod(bits, M).reshape(N_SC, N_SYM - 1)
            grid_tx = build_grid(data_syms, dmrs_seq)
            tx_sig = ofdm_mod(grid_tx, cp_lens=CP_LENS)
            # Channel + noise
            faded, h_time = ch(tx_sig)
            rx_sig, noise_var = awgn(faded, snr_db)
            # Receiver
            grid_rx = ofdm_demod(rx_sig, n_sc=N_SC, n_sym=N_SYM, cp_lens=CP_LENS)
            h_dmrs = ls_ce(grid_rx[:, DMRS_COL], dmrs_seq)
            h_hat = interp_freq(h_dmrs, np.arange(N_SC), N_SC)[:, None]
            data_cols = [c for c in range(N_SYM) if c != DMRS_COL]
            s_eq = zf_eq(grid_rx[:, data_cols], h_hat)
            bits_hat = qam_demod(s_eq, M)
            err += (bits_hat != bits).sum()
            tot += bits.size
        ber[i_snr] = err / tot
        print(f"SNR {snr_db:2d} dB → BER = {ber[i_snr]:.3e}")
    return snr_db_list, ber

run_sim()