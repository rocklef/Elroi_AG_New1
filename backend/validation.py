import pandas as pd
import torch
import torch.nn as nn
import joblib
import time
from datetime import timedelta
import os

# ---------------- Model Definition ---------------- #
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               padding=(3 - 1) * dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=(3 - 1) * dilation, dilation=dilation)
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
        x = x.unsqueeze(1)
        x = self.tcn(x)
        x = x[:, :, -1]
        return self.fc(x)


# ---------------- Real-Time Watcher ---------------- #
def wait_and_predict(excel_path, seq_len=240, target_temp=32.0):
    print("🔍 Watching for real-time data in:", excel_path)

    # Wait until Excel file has enough readings (20 min × 12 = 240)
    while True:
        if os.path.exists(excel_path):
            try:
                # ✅ Robust Excel reading (handles both .xls and .xlsx)
                if excel_path.endswith(".xls"):
                    df = pd.read_excel(excel_path, engine="xlrd")
                else:
                    df = pd.read_excel(excel_path, engine="openpyxl")

                if len(df) >= seq_len:
                    print(f"✅ Collected {len(df)} readings — starting prediction...")
                    break
            except Exception as e:
                print("⚠️ Error reading Excel:", e)
                time.sleep(10)
                continue
        else:
            print("⏳ Waiting for Excel file to appear...")
        time.sleep(10)  # check every 10 seconds

    # Process only last 20 minutes of data
    df = df.tail(seq_len)

    # --- Auto-detect columns (robust for any Excel format) ---
    time_col = next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()), None)
    temp_col = next((c for c in df.columns if 'temp' in c.lower()), None)

    if not time_col or not temp_col:
        raise ValueError(f"❌ Could not find time/temp columns in Excel! Found columns: {df.columns.tolist()}")

    df["time"] = pd.to_datetime(df[time_col])
    temps = df[temp_col].values.reshape(-1, 1)

    # Load scaler and trained model
    scaler = joblib.load("scaler.save")
    temps_scaled = scaler.transform(temps).flatten()

    model = HybridTCN()
    model.load_state_dict(torch.load("hybrid_tcn_dense.pth", map_location="cpu"))
    model.eval()

    # Predict future readings every 5 sec
    current_seq = list(temps_scaled)
    pred_times, pred_temps = [], []
    step = 0

    while True:
        x = torch.tensor(current_seq[-seq_len:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x).item()

        y_pred_real = scaler.inverse_transform([[y_pred]])[0, 0]
        pred_temps.append(y_pred_real)
        pred_times.append(df["time"].iloc[-1] + timedelta(seconds=5 * (step + 1)))
        current_seq.append(y_pred)
        step += 1

        if y_pred_real <= target_temp:
            print(f"✅ Reached predicted {target_temp}°C after {step * 5 / 60:.1f} minutes.")
            break

    # Save predicted values
    pred_df = pd.DataFrame({
        "time": [t.strftime("%d-%m-%Y %H:%M:%S") for t in pred_times],
        "temp": pred_temps
    })
    pred_df.to_csv("predicted_future.csv", index=False)
    print("📁 Predictions saved to predicted_future.csv")


if __name__ == "__main__":
    wait_and_predict(r"C:\Users\COE_AT Admin\Desktop\software\SRM PROJECT SUPPORT\NEW_CHANGE_24_10_25\Report_20251024H10.xls")
