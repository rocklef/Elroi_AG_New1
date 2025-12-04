import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import joblib

# ---------------- Dataset ---------------- #
class TempDataset(Dataset):
    def __init__(self, series, seq_length=30):
        self.seq_length = seq_length
        self.series = series

    def __len__(self):
        return len(self.series) - self.seq_length

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_length]
        y = self.series[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

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
        # Crop or pad to match residual
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
        x = x[:, :, -1]         # last timestep features
        return self.fc(x)

# ---------------- Training with LOOCV ---------------- #
SEQ_LEN = 30
FILES = [
    "Report_20250604-2.csv", "Report_20250605-1.csv", "Report_20250605-2.csv",
    "Report_20250611-1.csv", "Report_20250611-2.csv", "Report_20250611-3.csv",
    "Report_20250604-1.csv"
]

# Combine all data for scaler
all_temps = []
for f in FILES:
    df = pd.read_csv(f, parse_dates=["time"])
    all_temps.append(df["temp"].values.reshape(-1,1))
all_data = np.vstack(all_temps)

scaler = MinMaxScaler()
all_scaled = scaler.fit_transform(all_data).flatten()
joblib.dump(scaler, "scaler.save")

# LOOCV training
for test_file in FILES:
    print(f"\n--- Leaving out {test_file} for validation ---")
    train_files = [f for f in FILES if f != test_file]

    train_temps = []
    for f in train_files:
        df = pd.read_csv(f, parse_dates=["time"])
        train_temps.append(df["temp"].values.reshape(-1,1))
    train_data = np.vstack(train_temps)
    train_scaled = scaler.transform(train_data).flatten()

    dataset = TempDataset(train_scaled, seq_length=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = HybridTCN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        losses = []
        for x, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss={np.mean(losses):.6f}")

# Save final model
torch.save(model.state_dict(), "hybrid_tcn_dense.pth")
print("✅ Final model saved as hybrid_tcn_dense.pth")
