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
        # Downsample is the skip connection projection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        # Align lengths for the residual connection
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
        # Input shape: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        x = x.unsqueeze(1)
        x = self.tcn(x)
        # Take the feature vector corresponding to the last time step
        # Output shape: (batch_size, hidden_dim, 1) -> (batch_size, hidden_dim)
        x = x[:, :, -1]
        return self.fc(x)


# ---------------- Real-Time Watcher ---------------- #
def wait_and_predict(excel_path, seq_len=240, target_temp=32.0, poll_interval=10):
    print(f"🔍 Watching for real-time data in: {excel_path}")
    
    # Helper to read Excel safely
    def read_excel_safe(path):
        if path.endswith(".xls"):
            return pd.read_excel(path, engine="xlrd")
        else:
            return pd.read_excel(path, engine="openpyxl")

    # 1. Wait for file and record initial row count
    while True:
        if os.path.exists(excel_path):
            try:
                df_initial = read_excel_safe(excel_path)
                break
            except Exception as e:
                print(f"⚠️ Error reading Excel (initial check): {e}")
                time.sleep(poll_interval)
        else:
            print("⏳ Waiting for Excel file to appear...")
            time.sleep(poll_interval)

    initial_count = len(df_initial)
    print(f"✅ File found with {initial_count} existing rows. Waiting for {seq_len} new readings...")

    # 2. Wait until seq_len new rows are appended
    while True:
        try:
            df_now = read_excel_safe(excel_path)
        except Exception as e:
            print(f"⚠️ Error reading Excel (polling): {e}")
            time.sleep(poll_interval)
            continue

        current_count = len(df_now)
        new_rows_collected = current_count - initial_count
        
        if new_rows_collected >= seq_len:
            # Use only the last seq_len rows, which represent the new 20 minutes of data
            df = df_now.tail(seq_len).copy()
            print(f"✅ Collected {seq_len} new readings — starting prediction...")
            break
        else:
            print(f"⏳ Collected {new_rows_collected}/{seq_len} new readings. Waiting...")
            time.sleep(poll_interval)

    # --- Data Processing and Model Loading ---
    
    # Auto-detect columns
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

    # 3. Prediction Loop (Autoregressive)
    current_seq = list(temps_scaled)
    pred_times, pred_temps = [], []
    step = 0

    while True:
        # Prepare the last seq_len scaled values for the model input
        x = torch.tensor(current_seq[-seq_len:], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            y_pred = model(x).item()

        # Inverse transform the prediction back to the real temperature scale
        y_pred_real = scaler.inverse_transform([[y_pred]])[0, 0]
        
        # Record results
        pred_temps.append(y_pred_real)
        # Calculate the predicted time (5 seconds per step)
        predicted_time = df["time"].iloc[-1] + timedelta(seconds=5 * (step + 1))
        pred_times.append(predicted_time)
        
        # Add the *scaled* prediction back to the sequence for the next step (Autoregressive)
        current_seq.append(y_pred)
        step += 1
        
        # Check for target condition
        if y_pred_real <= target_temp:
            # --- MODIFIED OUTPUT HERE ---
            predicted_timestamp_str = predicted_time.strftime("%d-%m-%Y %H:%M:%S")
            time_to_reach = step * 5 / 60
            
            print(f"✅ Prediction Complete: Reached predicted {target_temp}°C in {time_to_reach:.1f} minutes.")
            print(f"   (Predicted Timestamp: {predicted_timestamp_str})")
            break

    # 4. Save predicted values
    pred_df = pd.DataFrame({
        "time": [t.strftime("%d-%m-%Y %H:%M:%S") for t in pred_times],
        "temp": pred_temps
    })
    pred_df.to_csv("predicted_future.csv", index=False)
    print("📁 Predictions saved to predicted_future.csv")


if __name__ == "__main__":
    # Use your specific file path
    excel_file_path = r"C:\Users\COE_AT Admin\Desktop\software\SRM PROJECT SUPPORT\NEW_CHANGE_24_10_25\Report_20251024H10.xls"
    wait_and_predict(excel_file_path)