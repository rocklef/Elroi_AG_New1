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
        # Align lengths
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
def wait_and_predict(excel_path, seq_len=240, target_temp=32.0, poll_interval=5):
    """
    Behaviour:
      - On start, record current number of rows in the Excel file.
      - Wait until seq_len new rows have been appended (len_now >= len_before + seq_len).
      - Use the last seq_len rows to run predictions (same as your original loop).
    """
    # Wait for file to exist and read initial row count
    while True:
        if os.path.exists(excel_path):
            try:
                if excel_path.endswith(".xls"):
                    df_initial = pd.read_excel(excel_path, engine="xlrd")
                else:
                    df_initial = pd.read_excel(excel_path, engine="openpyxl")
                break
            except Exception as e:
                print("[WARN] Error reading Excel (initial):", e)
                time.sleep(poll_interval)
        else:
            time.sleep(poll_interval)

    initial_count = len(df_initial)
    print(f"[OK] Initial row count: {initial_count}", flush=True)
    print(f"[WAIT] Waiting for {seq_len} new rows (target: {initial_count + seq_len})...", flush=True)

    # Wait until seq_len new rows are appended
    while True:
        try:
            if excel_path.endswith(".xls"):
                df_now = pd.read_excel(excel_path, engine="xlrd")
            else:
                df_now = pd.read_excel(excel_path, engine="openpyxl")
        except Exception as e:
            print(f"[WARN] Error reading: {e}", flush=True)
            time.sleep(poll_interval)
            continue

        current_count = len(df_now)
        if current_count >= initial_count + seq_len:
            # take the last seq_len rows (the new readings)
            df = df_now.tail(seq_len).copy()
            print(f"[OK] Got {seq_len} new rows! Starting prediction...", flush=True)
            break
        
        print(f"[INFO] Rows: {current_count}/{initial_count + seq_len} (need {initial_count + seq_len - current_count} more)", flush=True)
        time.sleep(poll_interval)

    # Detect columns
    time_col = next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()), None)
    temp_col = next((c for c in df.columns if 'temp' in c.lower()), None)

    if not time_col or not temp_col:
        raise ValueError(f"Could not find time/temp columns. Found: {df.columns.tolist()}")

    df["time"] = pd.to_datetime(df[time_col])
    temps = df[temp_col].values.reshape(-1, 1)

    # Load scaler and model
    scaler = joblib.load("scaler.save")
    temps_scaled = scaler.transform(temps).flatten()

    model = HybridTCN()
    model.load_state_dict(torch.load("hybrid_tcn_dense.pth", map_location="cpu"))
    model.eval()

    # Prediction loop (same semantics as your original code)
    current_seq = list(temps_scaled)
    pred_times, pred_temps = [], []
    step = 0

    print(f"[OK] Collected {seq_len} new readings — starting prediction...")

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
            predicted_minutes = step * 5 / 60
            print(f"[OK] Reached predicted {target_temp}°C after {predicted_minutes:.1f} minutes.")
            
            # Save prediction result as JSON for frontend
            import json
            result = {
                "status": "complete",
                "predicted_minutes": round(predicted_minutes, 1),
                "target_temp": target_temp,
                "timestamp": time.time()
            }
            with open("prediction_result.json", "w") as f:
                json.dump(result, f)
            print("[INFO] Prediction result saved to prediction_result.json")
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
