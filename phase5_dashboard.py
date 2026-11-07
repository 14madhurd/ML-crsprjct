import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import StandardScaler

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="XLR8@KT: Intelligent Dead Reckoning", layout="wide")
st.title("🚗 XLR8@KT: Edge-Deployable GNSS+INS Sensor Fusion Engine")
st.markdown("Simulating On-Device Smartphone Navigation during GNSS Outages")

# ==========================================
# MODEL DEFINITION
# ==========================================
class VelocityCNN(nn.Module):
    def __init__(self, num_features):
        super(VelocityCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.fc1 = nn.Linear(64 * 5, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1) 
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

# ==========================================
# LOAD DATA & MODEL (Cached)
# ==========================================
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('data/phase1_processed.csv')
    feature_cols = ['acc_x_aligned', 'acc_y_aligned', 'acc_z_aligned']
    speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()][0]
    
    X_raw = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    window_size = 20
    X_windows = [X_scaled[i : i + window_size].T for i in range(len(X_scaled) - window_size)]
    X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)
    
    actual_speed = df[speed_col].values[window_size - 1 : -1]
    return X_tensor, actual_speed

@st.cache_resource
def load_model():
    model = VelocityCNN(num_features=3)
    model.load_state_dict(torch.load('data/velocity_cnn.pth', weights_only=True))
    model.eval()
    return model

try:
    X_tensor, actual_speed = load_and_process_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading data or model weights. Make sure Phase 1 and 2 scripts ran successfully first! Details: {e}")
    st.stop()

with torch.no_grad():
    ai_speed = model(X_tensor).numpy()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Simulation Controls")
gnss_noise = st.sidebar.slider("GNSS Noise (Meters)", 0.5, 5.0, 2.0, step=0.5)
blackout_start = st.sidebar.number_input("Tunnel Entry (Seconds)", min_value=10, max_value=200, value=60)
blackout_end = st.sidebar.number_input("Tunnel Exit (Seconds)", min_value=50, max_value=300, value=150)

# ==========================================
# SENSOR FUSION LOGIC
# ==========================================
dt = 0.1 
time_steps = len(ai_speed)
true_position = np.cumsum(actual_speed * dt)

gnss_measurements = np.full(time_steps, np.nan)
for i in range(0, time_steps, 10):
    gnss_measurements[i] = true_position[i] + np.random.normal(0, gnss_noise)

start_idx, end_idx = int(blackout_start * 10), int(blackout_end * 10)
gnss_measurements[start_idx:end_idx] = np.nan 

kf = KalmanFilter(dim_x=1, dim_z=1)
kf.x = np.array([[0.]])
kf.P = np.array([[10.]])
kf.H = np.array([[1.]])
kf.R = np.array([[gnss_noise**2]])
kf.Q = np.array([[0.5]])

fused_positions = []
for i in range(time_steps):
    kf.x = kf.x + (ai_speed[i] * dt)
    kf.P = kf.P + kf.Q
    if not np.isnan(gnss_measurements[i]):
        kf.update(np.array([[gnss_measurements[i]]]))
    fused_positions.append(kf.x[0, 0])

# ==========================================
# DASHBOARD VISUALS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. AI Velocity Predictor (IMU Only)")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    time_axis = np.arange(3000) / 10.0
    ax1.plot(time_axis, actual_speed[:3000], label='Actual Ground Truth', color='black')
    ax1.plot(time_axis, ai_speed[:3000], label='AI Prediction', color='dodgerblue', linestyle='--')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (m/s)")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("2. GNSS+INS Fusion & Blackout Recovery")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    time_axis_full = np.arange(time_steps) * dt
    subset = min(2500, time_steps)
    
    ax2.plot(time_axis_full[:subset], true_position[:subset], color='black', label='True Trajectory')
    ax2.scatter(time_axis_full[:subset], gnss_measurements[:subset], color='orange', s=10, label='1Hz GNSS')
    ax2.plot(time_axis_full[:subset], fused_positions[:subset], color='blue', linestyle='--', label='Fused Output')
    ax2.axvspan(blackout_start, blackout_end, color='red', alpha=0.15, label='GNSS Blackout')
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Distance (m)")
    ax2.legend()
    st.pyplot(fig2)

# Metrics
st.markdown("---")
st.subheader("Performance Benchmarks")
if end_idx < len(true_position) and start_idx < len(true_position):
    blackout_distance = true_position[end_idx] - true_position[start_idx]
    fused_distance = fused_positions[end_idx] - fused_positions[start_idx]
    drift_error = abs(blackout_distance - fused_distance)
    drift_percentage = (drift_error / blackout_distance) * 100 if blackout_distance > 0 else 0
else:
    blackout_distance, drift_error, drift_percentage = 0, 0, 0

m1, m2, m3 = st.columns(3)
m1.metric("GNSS Denied Distance Traveled", f"{blackout_distance:.1f} meters")
m2.metric("Dead Reckoning Accumulated Drift", f"{drift_error:.1f} meters")
m3.metric("Drift Percentage (Goal <10%)", f"{drift_percentage:.2f}%")