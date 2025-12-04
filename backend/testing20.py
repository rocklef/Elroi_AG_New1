import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use("Agg")  # Save plots to file instead of showing window
import matplotlib.pyplot as plt

# ---------------- Model ---------------- #
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
        # match shapes
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

# ---------------- Testing ---------------- #
def test_model():
    # Load test CSV
    df = pd.read_csv("test.csv", parse_dates=["time"])
    times = df["time"]
    temps = df["temp"].values.reshape(-1, 1)

    # Load scaler
    scaler = joblib.load("scaler.save")
    temps_scaled = scaler.transform(temps).flatten()

    # Load trained model
    model = HybridTCN()
    model.load_state_dict(torch.load("hybrid_tcn_dense.pth", map_location="cpu"))
    model.eval()

    seq_length = 30
    observed = temps_scaled[:len(temps)]
    preds = []
    pred_times = []
    current_seq = list(observed[-seq_length:])

    # Predict until ~200 steps or until reaching ~31°C
    for step in range(200):
        x = torch.tensor(current_seq[-seq_length:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x).item()

        # scale back to real temperature
        y_pred_real = scaler.inverse_transform([[y_pred]])[0, 0]
        preds.append(y_pred_real)
        pred_times.append(times.iloc[-1] + pd.Timedelta(seconds=step + 1))
        current_seq.append(y_pred)

        if y_pred_real <= 31:  # stop at room temperature
            print(f"\n✅ Temperature expected to hit ~31°C at step {step+1}\n")
            break

    # Build combined cooling curve
    combined_times = list(times) + pred_times
    combined_temps = list(temps.flatten()) + preds

    # Save predictions to CSV
    out_df = pd.DataFrame({"time": combined_times, "temperature": combined_temps})
    out_df.to_csv("predictions.csv", index=False)
    print("📁 Predictions saved to predictions.csv")

    # Plot and save graph
    plt.figure(figsize=(12, 6))
    plt.plot(combined_times, combined_temps, label="Cooling Path", color="green")
    plt.axhline(31, color="red", linestyle="--", label="Room Temp ~31°C")
    plt.xlabel("Time (approx)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.title("Predicted Cooling Path After Test Data")
    plt.savefig("cooling_prediction.png", dpi=300)
    print("🖼️ Graph saved as cooling_prediction.png")

    # Final statement
    print("✅ Work completed successfully!")

if __name__ == "__main__":
    test_model()
