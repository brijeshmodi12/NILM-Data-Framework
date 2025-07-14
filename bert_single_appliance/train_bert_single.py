# For training BERT model for NILM single appliance
# Train from scratch or load model and run inference
# Input Data X:     Aggregate power of house [batch_size, seq_len] from multiple houses to train the model
#                   For Refit Dataset (sampled at 1/8 Hz), seq_len of 512 is used (approximately 68 minutes of data).
# Input Label Y:    Single appliance power [batch_size, seq_len]


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
from postprocessing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Dataset from Multiple Houses
def load_multiple_houses(house_ids, dataset_path, target_appliance, threshold_watts=20):
    all_X = []
    all_Y_single = []
    max_power = 0.0
    
    for house_id in house_ids:
        ptfile = os.path.join(dataset_path, f'h{house_id}_refit_tensor_dataset.pt')
        if not os.path.exists(ptfile):
            print(f"House {house_id} data not found.")
            continue

        data = torch.load(ptfile)
        X = data['X']  # shape: [N, seq_len]
        Y = data['Y']  # shape: [N, seq_len, num_appliances]
        appliance_labels = data['appliance_labels']
        # print(appliance_labels)

        if target_appliance not in appliance_labels:
            print(f"{target_appliance} not in appliance labels for house {house_id}")
            continue

        
        idx = appliance_labels.index(target_appliance)
        Y_single = Y[:, :, idx]

        # house_max = float(Y_single.max())
        house_max = torch.quantile(Y_single.flatten(), 0.99)

        max_power = max(max_power, house_max)

        if max_power == house_max:
            print(f'Max power from House {house_id}')

        all_X.append(X)
        all_Y_single.append(Y_single)

    if not all_X:
        raise ValueError("No valid data found.")

    # === Concatenate across houses
    X_combined = torch.cat(all_X, dim=0)
    Y_combined = torch.cat(all_Y_single, dim=0)

    # max_power = torch.tensor([3261.09])

    # === Normalize using global max power
    X_combined = X_combined / max_power
    Y_combined = Y_combined / max_power

    X_combined = torch.clamp(X_combined, 0.0, 1.0)
    Y_combined = torch.clamp(Y_combined, 0.0, 1.0)

    # === Mask sequences with enough activity
    threshold = threshold_watts / max_power
    mask = (Y_combined > threshold).sum(dim=1) > 5
    X_final = X_combined[mask]
    Y_final = Y_combined[mask]
   

    print(f"Loaded from {len(house_ids)} houses")
    print(f"Total samples after masking: {X_final.shape[0]}")

    return X_final, Y_final, max_power

dataset_path = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\refit_clean"
house_list = range(1,10)
target_appliance = 'washing_machine' 
X, Y_single, MAX_POWER = load_multiple_houses(house_list, dataset_path, target_appliance)
MAX_POWER = MAX_POWER.item()
print(MAX_POWER)
# sys.exit()

# Training Parameters
epochs = 30
batch_size = 64
learning_rate = 1e-4

# Set load_model to False to train the network from scratch.
load_model = True
if load_model:
    model_path = f"best_model_{target_appliance}.pth"
    print('Setting Model Path for Preloading...')

# === Dataset & Dataloaders ===
full_dataset = TensorDataset(X, Y_single)
n_total = len(full_dataset)
n_train = int(0.1 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
)

# Determine if GPU is available
pin = torch.cuda.is_available()

# Optimized DataLoader settings
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin, persistent_workers=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=pin, persistent_workers=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=pin, persistent_workers=False)

# Model
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=1024, d_model=128, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)].to(x.device))

class BERT4NILM(nn.Module):
    def __init__(self, seq_len, d_model=128, n_head=8, n_layers=4, d_feedforward=256, dropout=0.1, cnn_channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, cnn_channels, kernel_size=7, padding=3)
        self.cnn_activation = nn.ReLU()
        self.embedding = nn.Linear(cnn_channels, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(max_len=seq_len, d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)           # [B, 1, L]
        x = self.conv1(x)            # [B, C, L]
        x = self.cnn_activation(x)
        x = x.permute(0, 2, 1)       # [B, L, C]
        x = self.embedding(x)        # [B, L, d_model]
        x = self.input_norm(x)
        x = self.dropout(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x_decoded = self.decoder(x)           # [B, L, 1]
        return x_decoded 
    
# Train  / Load Model
model = BERT4NILM(seq_len=512).to(device)

if load_model:
    model.load_state_dict(torch.load(model_path)) 
    print('Model Loaded! Training Skipped!')
else:
    print('Training Model...')

    def weighted_l1_loss(pred, target, threshold=0.05):
        weights = torch.where(target > threshold, 2.0, 1.0)
        return (weights * (pred - target).abs()).mean()

    criterion = lambda pred, y: weighted_l1_loss(pred, y, threshold=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(-1)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(-1)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Save the Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"best_model_{target_appliance}.pth")
            print(f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.6f}")

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

# === Test & Evaluation ===

model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        pred = model(x)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.unsqueeze(-1).numpy())


all_preds = np.concatenate(all_preds, axis=0).squeeze(-1) * MAX_POWER
all_targets = np.concatenate(all_targets, axis=0).squeeze(-1) * MAX_POWER

print(all_preds.shape)
print(all_targets.shape)

# === User-defined sample range
num_samples_to_plot = 200     # Change this to any number
start_index = 400             # Starting test sample index

# === Concatenate sequences
gt_concat = np.concatenate(all_targets[start_index:start_index + num_samples_to_plot], axis=0)
pred_concat = np.concatenate(all_preds[start_index:start_index + num_samples_to_plot], axis=0)

# === Time axis in hours
sampling_interval = 8  # seconds
time_hours = np.arange(len(gt_concat)) * sampling_interval / 3600  # convert to hours

# === Plot
plt.figure(figsize=(14, 4))
plt.plot(time_hours, gt_concat, label="Ground Truth")
plt.plot(time_hours, pred_concat, label='Prediction')
plt.title(f"{target_appliance.title()} - Samples {start_index} to {start_index + num_samples_to_plot - 1}")
plt.xlabel("Time (hours)")
plt.ylabel("Power (W)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Clip negatives
all_preds = np.clip(all_preds, 0, None)

mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
rmse = root_mean_squared_error(all_targets.flatten(), all_preds.flatten())
print(f"\nAppliance: {target_appliance}")
print(f"Test MAE: {mae:.2f} W")
print(f"Test RMSE: {rmse:.2f} W")


threshold_watts = 200

# Compute binary labels
gt_bin = get_binary_predictions(all_targets, threshold_watts=threshold_watts)
pred_bin = get_binary_predictions(all_preds, threshold_watts=threshold_watts)

# # Step 2: Post-process only the predictions
# pred_bin = smooth_predictions(pred_bin, window_seconds=40, sample_interval=8)
# pred_bin = filter_short_ons(pred_bin, min_duration_seconds=80, sample_interval=8)

gt_concat = np.concatenate(gt_bin[start_index:start_index + num_samples_to_plot], axis=0)
pred_concat = np.concatenate(pred_bin[start_index:start_index + num_samples_to_plot], axis=0)


# === Plot
plt.figure(figsize=(14, 4))
plt.plot(time_hours, gt_concat, label="Ground Truth")
plt.plot(time_hours, pred_concat, label='Prediction')
plt.title(f"{target_appliance.title()} - Samples {start_index} to {start_index + num_samples_to_plot - 1}")
plt.xlabel("Time (hours)")
plt.ylabel("Power (W)")
plt.grid(True)
plt.ylim(0,2)
plt.legend()
plt.tight_layout()
plt.show()


print(gt_bin.shape)
print(pred_bin.shape)

# sys.exit()


gt_bin = gt_bin.flatten()
pred_bin = pred_bin.flatten()
precision = precision_score(gt_bin, pred_bin)
recall = recall_score(gt_bin, pred_bin)
f1 = f1_score(gt_bin, pred_bin)
accuracy = accuracy_score(gt_bin, pred_bin)

print("\nBinary Evaluation (Threshold = {} W)".format(threshold_watts))
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"Accuracy:  {accuracy:.3f}")

cm = confusion_matrix(gt_bin, pred_bin, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Off', 'On'])
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix (Threshold = {threshold_watts}W)")
plt.show()


# Use predicted power directly as a score
# precisions, recalls, thresholds = precision_recall_curve(gt_bin, all_preds.flatten())
all_preds = all_preds.flatten()
# Find closest threshold to 200W
precision, recall, thresholds = precision_recall_curve(gt_bin, all_preds)

# Append large value to thresholds to match recall/precision length
thresholds = np.append(thresholds, [all_preds.max() + 1])

# Find index closest to 200W
threshold_200_idx = np.argmin(np.abs(thresholds - 200))

# Plot PR curve
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label="PR Curve", color='blue')

# Mark the 200W threshold point
plt.scatter(recall[threshold_200_idx], precision[threshold_200_idx],
            color='red', label='200W Threshold', zorder=5)

# Labels and legend
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


