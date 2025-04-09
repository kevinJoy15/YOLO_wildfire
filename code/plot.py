import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files (update paths if needed)
vanilla = pd.read_csv('vanilla.csv')
cbam = pd.read_csv('CBAM_conv.csv')

# Function to extract metrics
def extract_metrics(df, model_name):
    return {
        'epoch': df['epoch'],
        'precision': df['metrics/precision(B)'],
        'recall': df['metrics/recall(B)'],
        'mAP50': df['metrics/mAP50(B)'],
        'model': model_name
    }

# Extract metrics for Vanilla and CBAM
metrics_vanilla = extract_metrics(vanilla, 'Vanilla')
metrics_cbam = extract_metrics(cbam, 'CBAM')

# Combine into a single DataFrame
comparison_df = pd.concat([
    pd.DataFrame(metrics_vanilla),
    pd.DataFrame(metrics_cbam)
])

# Plot settings
plt.figure(figsize=(15, 5))
plt.suptitle('CBAM vs. Vanilla Model Comparison', fontsize=14)

# 1. Precision
plt.subplot(1, 3, 1)
for model in ['Vanilla', 'CBAM']:
    data = comparison_df[comparison_df['model'] == model]
    plt.plot(data['epoch'], data['precision'], label=model, linewidth=2)
plt.title('Precision (B)')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.legend()

# 2. Recall
plt.subplot(1, 3, 2)
for model in ['Vanilla', 'CBAM']:
    data = comparison_df[comparison_df['model'] == model]
    plt.plot(data['epoch'], data['recall'], label=model, linewidth=2)
plt.title('Recall (B)')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.legend()

# 3. mAP50
plt.subplot(1, 3, 3)
for model in ['Vanilla', 'CBAM']:
    data = comparison_df[comparison_df['model'] == model]
    plt.plot(data['epoch'], data['mAP50'], label=model, linewidth=2)
plt.title('mAP50 (B)')
plt.xlabel('Epoch')
plt.ylabel('mAP50')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()