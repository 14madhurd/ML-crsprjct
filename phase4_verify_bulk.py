import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
from scipy.signal import butter, filtfilt

# ==========================================
# 1. ANDROID-STYLE PREPROCESSING FUNCTIONS
# ==========================================
def apply_lowpass_filter(data, cutoff_freq=2.0, sample_rate=10.0, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data.fillna(0))

def apply_virtual_alignment(df):
    speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()]
    stationary_data = df[df[speed_col[0]] < 0.5] if speed_col else df.head(50)
    if stationary_data.empty: stationary_data = df.head(50)
        
    acc_cols = [col for col in df.columns if 'acc' in col.lower() and 'clean' in col.lower()]
    g_x, g_y, g_z = stationary_data[acc_cols[0]].mean(), stationary_data[acc_cols[1]].mean(), stationary_data[acc_cols[2]].mean()
    
    pitch = np.arctan2(-g_x, np.sqrt(g_y**2 + g_z**2))
    roll = np.arctan2(g_y, g_z)
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R = np.dot(Ry, Rx)
    
    acc_matrix = df[[acc_cols[0], acc_cols[1], acc_cols[2]]].values
    aligned_acc = np.dot(acc_matrix, R.T)
    df['acc_x_aligned'], df['acc_y_aligned'], df['acc_z_aligned'] = aligned_acc[:, 0], aligned_acc[:, 1], aligned_acc[:, 2]
    return df

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
# 2. VERIFICATION PIPELINE
# ==========================================
if __name__ == "__main__":
    # ---> CHANGE THESE TO TEST DIFFERENT DRIVES <---
    s_test_file = 'data/S-M.csv' 
    v_test_file = 'data/V-M.csv'
    
    print(f"Loading Test Data: {s_test_file} ...")
    df_s = pd.read_csv(s_test_file, encoding='latin1')
    df_v = pd.read_csv(v_test_file, encoding='latin1')
    
    # Sync
    time_col_s = [col for col in df_s.columns if 'time' in col.lower()][0]
    time_col_v = [col for col in df_v.columns if 'time' in col.lower()][0]
    df_s = df_s.sort_values(time_col_s).rename(columns={time_col_s: 'time'})
    df_v = df_v.sort_values(time_col_v).rename(columns={time_col_v: 'time'})
    df_s['time'] = df_s['time'].astype(float)
    df_v['time'] = df_v['time'].astype(float)
    df = pd.merge_asof(df_s, df_v, on='time', direction='nearest')
    
    # Clean & Align
    raw_cols = [col for col in df.columns if 'acc' in col.lower() and 'clean' not in col.lower()]
    for col in raw_cols: df[f'{col}_clean'] = apply_lowpass_filter(df[col])
    df = apply_virtual_alignment(df)
    
    # Load Global Scaler (Simulating Android)
    print("Loading global scaler parameters...")
    with open('data/scaler_params.json', 'r') as f:
        scaler_params = json.load(f)
    mean_vals = np.array(scaler_params['mean'])
    scale_vals = np.array(scaler_params['scale'])
    
    # Apply Scaling
    feature_cols = ['acc_x_aligned', 'acc_y_aligned', 'acc_z_aligned']
    speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()][0]
    
    X_raw = df[feature_cols].values
    X_scaled = (X_raw - mean_vals) / scale_vals
    
    # Create Windows
    window_size = 20
    X_windows = [X_scaled[i : i + window_size].T for i in range(len(X_scaled) - window_size)]
    X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)
    actual_speed = df[speed_col].values[window_size - 1 : -1]
    
    # Load Bulk Model
    print("Loading Bulk-Trained Model...")
    model = VelocityCNN(num_features=3)
    model.load_state_dict(torch.load('data/bulk_velocity_cnn.pth', weights_only=True))
    model.eval()
    
    print("Running Inference...")
    with torch.no_grad():
        ai_speed = model(X_tensor).numpy()
        
    # Plot Results
    print("Generating Verification Plot...")
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(ai_speed)) / 10.0
    
    # Plot just the first 500 seconds so we can see the detail
    subset = 5000 
    plt.plot(time_axis[:subset], actual_speed[:subset], color='black', linewidth=2.5, label='Actual CAN Speed (Ground Truth)')
    plt.plot(time_axis[:subset], ai_speed[:subset], color='dodgerblue', linestyle='--', linewidth=1.5, label='AI Predicted Speed (Bulk Model)')
    
    plt.title(f"Bulk Model Verification Test on {s_test_file}")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    out_img = 'data/bulk_verification_plot.png'
    plt.savefig(out_img, dpi=300)
    print(f"Saved to {out_img}!")
    plt.show()