import pandas as pd
import matplotlib.pyplot as plt

# Load version 6 metrics
df = pd.read_csv("simclr_logs/version_6/metrics.csv")

# Filter epoch-level metrics
epoch_metrics = df[df['epoch'].notna()]

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(epoch_metrics['epoch'], epoch_metrics['val_loss'], 'b-o')
plt.title('Validation Loss')
plt.subplot(122)
plt.plot(epoch_metrics['epoch'], epoch_metrics['val_acc_top1'], 'r-o')
plt.title('Top-1 Accuracy')
plt.show()