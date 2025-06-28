import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv('clas_loss_acc.csv')

# Đổi tên cột cho nhất quán (nếu cần)
df.columns = df.columns.str.lower()  # biến 'Epoch' → 'epoch', 'Accuracy' → 'accuracy'

# Làm mượt bằng trung bình trượt
window_size = 5
df['accuracy_smooth'] = df['accuracy'].rolling(window=window_size, min_periods=1).mean()

# Vẽ
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['accuracy'], label='Original Accuracy', alpha=0.3, linestyle='--', color='gray')
plt.plot(df['epoch'], df['accuracy_smooth'], label=f'Smoothed (window={window_size})', color='blue')

plt.title('Accuracy theo Epoch (Smoothed)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
