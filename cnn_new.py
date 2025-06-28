import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
from no_lib import OFDM

pilot_col = 3

# === 1. Mô hình CNN dự đoán 16 class (QAM16) ===
class OFDM_ClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)  # 16 class output
        )

    def forward(self, x):
        return self.net(x)  # output: (batch, 16, 3276, 14)

# === 2. Hàm ánh xạ complex QAM → label 0–15 ===
def qam16_label_mapper(symbols):
    mapping = [(-3 - 3j), (-3 - 1j), (-3 + 1j), (-3 + 3j),
               (-1 - 3j), (-1 - 1j), (-1 + 1j), (-1 + 3j),
               (1 - 3j), (1 - 1j), (1 + 1j), (1 + 3j),
               (3 - 3j), (3 - 1j), (3 + 1j), (3 + 3j)]
    mapping = np.array(mapping) / np.sqrt(10)
    flattened = symbols.flatten()
    labels = np.zeros(flattened.shape, dtype=np.int64)
    for i, s in enumerate(flattened):
        dists = np.abs(s - mapping)
        labels[i] = np.argmin(dists)
    return labels.reshape(symbols.shape)

# === 3. Chuyển complex → 2 channel tensor ===
def complex_to_channels(x):
    return torch.tensor(np.stack([np.real(x), np.imag(x)], axis=0), dtype=torch.float32)

# === 4. Sinh data: input_tensor, label_tensor ===
def get_classification_sample(ofdm):
    rx, clean_grid, _, _ = ofdm.dieu_che_OFDM()
    rx_grid = ofdm.extract_sybols(rx)
    labels = qam16_label_mapper(clean_grid)  # shape (3276,14)
    input_tensor = complex_to_channels(rx_grid)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return input_tensor, label_tensor

# === 5. Huấn luyện với checkpoint và log CSV ===
def train(model, ofdm, epochs=500, batch_size=4, resume=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    log_path = 'clas_loss_acc.csv'
    ckpt_latest = 'checkpoint_latest_class.pth'

    # Resume nếu có checkpoint
    if resume and os.path.exists(ckpt_latest):
        ckpt = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"🔁 Resumed from epoch {start_epoch}")

    # Ghi header CSV nếu lần đầu
    if not os.path.exists(log_path) or start_epoch == 0:
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Loss', 'Accuracy'])

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        total_correct = 0
        total_symbol = 0

        model.train()
        for _ in range(batch_size):
            inp, label = get_classification_sample(ofdm)
            inp = inp.unsqueeze(0).to(device)      # (1,2,3276,14)
            label = label.unsqueeze(0).to(device)  # (1,3276,14)

            optimizer.zero_grad()
            logits = model(inp)  # (1,16,3276,14)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_correct += (pred == label).sum().item()
            total_symbol += label.numel()

        avg_loss = epoch_loss / batch_size
        accuracy = total_correct / total_symbol
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | Acc: {accuracy:.4f}")

        # Lưu log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, accuracy])

        # Lưu checkpoint mỗi 100 epoch
        if (epoch+1) % 100 == 0:
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(ckpt_data, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(ckpt_data, ckpt_latest)
            print(f"💾 Checkpoint saved at epoch {epoch+1}")

    # Cuối training: lưu final
    torch.save({'epoch': epochs-1,
                'model_state_dict': model.state_dict()}, 'classifier_cnn.pth')
    print('✅ Final model saved to classifier_cnn.pth')

# === 6. Test sau train ===
if __name__ == '__main__':
    model = OFDM_ClassifierCNN()
    ofdm = OFDM()
    train(model, ofdm, epochs=8100, batch_size=4, resume=True)

    # Test trên 1 sample mới
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        rx, clean_grid, _, bits = ofdm.dieu_che_OFDM()
        rx_grid = ofdm.extract_sybols(rx)
        inp = complex_to_channels(rx_grid).unsqueeze(0).to(device)

        logits = model(inp)
        pred_lbl = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (3276,14)

        # Chuyển label → symbol
        qam_map = np.array([(-3 - 3j),(-3 - 1j),(-3 + 1j),(-3 + 3j),
                             (-1 - 3j),(-1 - 1j),(-1 + 1j),(-1 + 3j),
                             (1 - 3j),(1 - 1j),(1 + 1j),(1 + 3j),
                             (3 - 3j),(3 - 1j),(3 + 1j),(3 + 3j)]) / np.sqrt(10)
        pred_sym = qam_map[pred_lbl]

        # Gộp bỏ cột pilot
        cols = ofdm.symbols
        pc = ofdm.pilot_col
        rx_sym = np.concatenate([pred_sym[:, c] for c in range(cols) if c != pc])

        # Demod về bits và tính BER
        rx_bits = ofdm.demod16(rx_sym)
        ber = np.mean(bits != rx_bits)

    print(f"\n=== Test kết thúc ===")
    print(f"BER trên sample mới: {ber:.6f}")
