import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use("Agg")  # Save plots instead of showing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------- Model ---------------- #
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               padding=(3-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=(3-1)*dilation, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if out.size(2) > residual.size(2):
            out = out[:, :, :residual.size(2)]
        elif out.size(2) < residual.size(2):
            residual = residual[:, :, :out.size(2)]
        return self.relu(out + residual)

class HybridTCN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=5):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(ResidualBlock(input_dim if i == 0 else hidden_dim, hidden_dim, dilation))
        self.tcn = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)      # [B,1,seq_len]
        x = self.tcn(x)
        x = x[:, :, -1]
        return self.fc(x)

# ---------------- Testing LOOCV ---------------- #
SEQ_LEN = 30
FILES = [
    "Report_20250604-2.csv", "Report_20250605-1.csv", "Report_20250605-2.csv",
    "Report_20250611-1.csv", "Report_20250611-2.csv", "Report_20250611-3.csv",
    "Report_20250604-1.csv"
]

# Load scaler and model
scaler = joblib.load("scaler.save")
model = HybridTCN()
model.load_state_dict(torch.load("hybrid_tcn_dense.pth", map_location="cpu"))
model.eval()

for test_file in FILES:
    print(f"\n--- Testing on {test_file} ---")
    df_test = pd.read_csv(test_file, parse_dates=["time"])
    temps = df_test["temp"].values.reshape(-1, 1)
    temps_scaled = scaler.transform(temps).flatten()
    
    current_seq = list(temps_scaled[:SEQ_LEN])
    preds_scaled = []

    # Predict for rest of the series
    for i in range(SEQ_LEN, len(temps_scaled)):
        x = torch.tensor(current_seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x).item()
        preds_scaled.append(y_pred)
        current_seq.append(temps_scaled[i])  # Use actual for LOOCV testing

    # Rescale predictions to original temperature
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    actual = temps[SEQ_LEN:].flatten()
    times = df_test["time"].iloc[SEQ_LEN:]

    # Accuracy metrics
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Save CSV of actual vs predicted
    out_df = pd.DataFrame({"time": times, "actual_temp": actual, "predicted_temp": preds})
    csv_name = f"LOOCV_pred_{os.path.basename(test_file)}"
    out_df.to_csv(csv_name, index=False)
    print(f"CSV saved as {csv_name}")

    # Plot graph
    plt.figure(figsize=(12,6))
    plt.plot(times, actual, label="Actual", color="blue")
    plt.plot(times, preds, label="Predicted", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.title(f"LOOCV Test: {test_file}")
    plt.legend()
    plt.tight_layout()
    graph_name = f"LOOCV_graph_{os.path.basename(test_file).replace('.csv','.png')}"
    plt.savefig(graph_name, dpi=300)
    plt.close()
    print(f"Graph saved as {graph_name}")
