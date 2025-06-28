import numpy as np
import pandas as pd
from numpy.fft import fft, ifft
from scipy.signal import convolve

# --- Tham số ---
N = 4096
subc = 3276
symbols = 14
pilot_col = 3
CP0, CP = 352, 288
SNR_dB = 20
padLeft = (N - subc) // 2
padRight = N - subc - padLeft
fd = 25
Ts_ns = 8.14

# Đọc DMRS từ file Excel
try:
    dmrs_data = pd.read_excel("l1_mini_project_ce_eq_dmrs_data.xlsx",
                              sheet_name="Sheet1", usecols="A", header=None, engine='openpyxl')
    dmrs_values = dmrs_data.dropna().values.flatten()
    if len(dmrs_values) != subc:
        raise ValueError(f"Số lượng DMRS đọc từ Excel ({len(dmrs_values)}) không khớp với số subcarrier ({subc})")
    dmrs_cleaned = [str(x).replace('i', 'j') for x in dmrs_values]
    dmrs = np.array(dmrs_cleaned, dtype=complex)
except Exception as e:
    raise RuntimeError(f"Không thể đọc DMRS từ file Excel: {e}")


class OFDM:
    def __init__(self, N=N, subc=subc, symbols=symbols, pilot_col=pilot_col, CP0=CP0, CP=CP):
        self.N = N
        self.subc = subc
        self.symbols = symbols
        self.pilot_col = pilot_col
        self.CP0 = CP0
        self.CP = CP

    def qam16(self, b):
        m = {}
        mapping = [(-3 - 3j), (-3 - 1j), (-3 + 1j), (-3 + 3j),
                   (-1 - 3j), (-1 - 1j), (-1 + 1j), (-1 + 3j),
                   (1 - 3j), (1 - 1j), (1 + 1j), (1 + 3j),
                   (3 - 3j), (3 - 1j), (3 + 1j), (3 + 3j)]
        for i, bb in enumerate([(a, b, c, d) for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]]):
            m[bb] = mapping[i] / np.sqrt(10)
        arr = b.reshape(-1, 4)
        return np.array([m[tuple(r)] for r in arr])

    def demod16(self, symbols):
        mapping = [(-3 - 3j), (-3 - 1j), (-3 + 1j), (-3 + 3j),
                   (-1 - 3j), (-1 - 1j), (-1 + 1j), (-1 + 3j),
                   (1 - 3j), (1 - 1j), (1 + 1j), (1 + 3j),
                   (3 - 3j), (3 - 1j), (3 + 1j), (3 + 3j)]
        bit_patterns = [(a, b, c, d) for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]]
        reverse_map = {mapping[i] / np.sqrt(10): bit_patterns[i] for i in range(16)}

        demod_bits = []
        for s in symbols:
            closest = min(reverse_map.keys(), key=lambda x: abs(s - x))
            demod_bits.extend(reverse_map[closest])
        return np.array(demod_bits)

    def add_TDL(self, tx):
        tap_delays_ns = [0, 10, 20, 30, 35, 45, 55, 120, 170, 245, 330, 480]
        tap_powers_db = [0, -2.2, -0.6, -0.6, -0.3, -1.2, -5.9, -2.2, -0.8, -6.3, -7.5, -7.1]
        Ts = Ts_ns * 1e-9
        tap_indices = [int(np.floor(delay / Ts_ns)) for delay in tap_delays_ns]
        max_index = max(tap_indices)
        tap_powers_linear = 10 ** (np.array(tap_powers_db) / 10)

        np.random.seed(42)
        tap_coeffs = np.sqrt(tap_powers_linear / 2) * (np.random.randn(12) + 1j * np.random.randn(12))

        h = np.zeros(max_index + 1, dtype=complex)
        for idx, power in zip(tap_indices, tap_powers_linear):
            amp = np.sqrt(power / 2)
            phi = 2 * np.pi * np.random.rand()
            f_d = np.random.uniform(-fd, fd)
            fading = amp * np.exp(1j * (2 * np.pi * f_d * Ts * idx + phi))
            h[idx] = fading
        rx = convolve(tx, h, mode='full')[:len(tx)]
        return rx

    def add_noise(self, rx):
        Ps = np.mean(np.abs(rx) ** 2)
        Pn = Ps / (10 ** (SNR_dB / 10))
        noise = np.sqrt(Pn / 2) * (np.random.randn(len(rx)) + 1j * np.random.randn(len(rx)))
        return rx + noise

    def extract_sybols(self, rx):
        idx = 0
        rx_grid_zero = np.zeros((N, symbols), dtype=complex)
        for col in range(symbols):
            cpL = CP0 if col == 0 else CP
            block = rx[idx + cpL:idx + cpL + N]
            Y = fft(block, N)[:N]
            rx_grid_zero[:, col] = Y
            idx += cpL + N

        rx_grid = np.zeros((subc, symbols), dtype=complex)
        for col in range(symbols):
            rx_grid[:, col] = rx_grid_zero[padLeft:-padRight, col]
        return rx_grid

    def dieu_che_OFDM(self):
        bits = np.random.randint(0, 2, subc * (symbols - 1) * 4)
        symb_q = self.qam16(bits)

        grid = np.zeros((N, symbols), dtype=complex)
        idx = 0
        for col in range(symbols):
            if col == pilot_col:
                grid[padLeft:-padRight, col] = dmrs
            else:
                grid[padLeft:-padRight, col] = symb_q[idx:idx + subc]
                idx += subc

        tx = []
        for col in range(symbols):
            x = ifft(grid[:, col], N)
            cp = x[-CP0:] if col == 0 else x[-CP:]
            tx.append(np.concatenate([cp, x]))
        tx = np.concatenate(tx)

        rx = self.add_TDL(tx)
        rx = self.add_noise(rx)

        return rx, grid[padLeft:-padRight, :], symb_q, bits

    def giai_dieu_che_OFDM(self, rx):
        rx_grid = self.extract_sybols(rx)

        estH = np.zeros((subc, symbols), dtype=complex)
        H_col = np.zeros(subc, dtype=complex)
        for i in range(subc):
            if dmrs[i] != 0:
                H_col[i] = rx_grid[i, pilot_col] / dmrs[i]
            else:
                H_col[i] = H_col[i - 1] if i > 0 else 1  # Giữ giá trị gần nhất hoặc mặc định 1
        for c in range(symbols):
            estH[:, c] = H_col

        rx_grid = rx_grid / estH
        rx_sym = np.concatenate([rx_grid[:, c] for c in range(symbols) if c != pilot_col])
        rx_bits = self.demod16(rx_sym)
        return rx_bits, rx_sym


if __name__ == "__main__":
    ofdm = OFDM()
    rx, grid, symb_q, bits = ofdm.dieu_che_OFDM()
    rx_bits, rx_sym = ofdm.giai_dieu_che_OFDM(rx)
    mse = np.mean(np.abs(symb_q - rx_sym) ** 2)
    ber = np.mean(bits != rx_bits)
    print("MSE=", mse, "BER=", ber)
