import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

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
        x = x.unsqueeze(1)
        x = self.tcn(x)
        x = x[:, :, -1]
        return self.fc(x)

# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="Cooling Prediction - Elroi Automation", layout="centered")

st.image("companylogo.jpeg", width=200)
st.markdown("<h1 style='text-align: center; color: #0077b6;'>Elroi Automation Private Limited</h1>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("### Upload CSV File")
st.info("Upload a CSV with 'time' and 'temp' columns (approx 20 min data) to predict cooling to 35°C.")

uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["time"])
    st.markdown("### Uploaded Data Preview")
    st.dataframe(df.head())

    st.markdown("### Time vs Temperature Graph")
    st.line_chart(df.set_index("time")["temp"])

    # ---------------- Prediction ---------------- #
    SEQ_LEN = 30
    THRESH_TARGET = 35.0

    # Load scaler and model
    scaler = joblib.load("scaler.save")
    model = HybridTCN()
    model.load_state_dict(torch.load("hybrid_tcn_dense.pth", map_location="cpu"))
    model.eval()

    temps = df["temp"].values.reshape(-1, 1)
    temps_scaled = scaler.transform(temps).flatten()
    current_seq = list(temps_scaled[-SEQ_LEN:])
    preds = []
    pred_times = []

    for step in range(2000):  # large enough steps
        x = torch.tensor(current_seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x).item()
        y_pred_real = scaler.inverse_transform([[y_pred]])[0, 0]
        preds.append(y_pred_real)
        pred_times.append(df["time"].iloc[-1] + timedelta(seconds=step+1))
        current_seq.append(y_pred)
        if y_pred_real <= THRESH_TARGET:
            final_time = df["time"].iloc[-1] + timedelta(seconds=step+1)
            break

    # Combine original + predicted
    combined_times = list(df["time"]) + pred_times
    combined_temps = list(df["temp"]) + preds

    # Plot in Streamlit
    st.markdown("### Predicted Cooling Path")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(combined_times, combined_temps, label="Predicted Cooling", color="green")
    ax.axhline(THRESH_TARGET, color="red", linestyle="--", label=f"Target Temp {THRESH_TARGET}°C")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

    # Display final timestamp
    st.subheader("Prediction Result")
    st.success(f"**Predicted time to reach {THRESH_TARGET}°C:** {final_time}")

    # Optional: Save CSV & graph
    out_df = pd.DataFrame({"time": combined_times, "temperature": combined_temps})
    out_df.to_csv("predictions_streamlit.csv", index=False)
    fig.savefig("cooling_prediction_streamlit.png", dpi=300)
    st.info("✅ Prediction CSV and graph saved successfully.")
