import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from scipy.signal import butter, filtfilt

def apply_lowpass_filter(data, cutoff_freq=2.0, sample_rate=10.0, order=4):
    b, a = butter(order, (cutoff_freq / (0.5 * sample_rate)), btype='low', analog=False)
    return filtfilt(b, a, data.fillna(0))

def apply_virtual_alignment(df):
    speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()]
    stationary_data = df[df[speed_col[0]] < 0.5] if speed_col else df.head(50)
    if stationary_data.empty: stationary_data = df.head(50)
    acc_cols = [col for col in df.columns if 'acc' in col.lower() and 'clean' in col.lower()]
    g_x, g_y, g_z = stationary_data[acc_cols[0]].mean(), stationary_data[acc_cols[1]].mean(), stationary_data[acc_cols[2]].mean()
    pitch, roll = np.arctan2(-g_x, np.sqrt(g_y**2 + g_z**2)), np.arctan2(g_y, g_z)
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    aligned_acc = np.dot(df[[acc_cols[0], acc_cols[1], acc_cols[2]]].values, np.dot(Ry, Rx).T)
    df['acc_x_aligned'], df['acc_y_aligned'], df['acc_z_aligned'] = aligned_acc[:, 0], aligned_acc[:, 1], aligned_acc[:, 2]
    return df

# ==========================================
# MUST MATCH DUAL-HEAD ARCHITECTURE
# ==========================================
class DualHeadVelocityCNN(nn.Module):
    def __init__(self, num_features):
        super(DualHeadVelocityCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        
        self.speed_fc1 = nn.Linear(256 * 12, 128)
        self.speed_drop = nn.Dropout(0.4)
        self.speed_out = nn.Linear(128, 1)
        
        self.dir_fc1 = nn.Linear(256 * 12, 128)
        self.dir_drop = nn.Dropout(0.4)
        self.dir_out = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        features = self.flatten(x)
        speed = self.speed_out(torch.relu(self.speed_drop(self.speed_fc1(features))))
        direction = self.dir_out(torch.relu(self.dir_drop(self.dir_fc1(features))))
        return speed.squeeze(), direction.squeeze()

if __name__ == "__main__":
    # ---> USE YOUR UNSEEN HOLDOUT FILES HERE <---
    s_test_file = 'data/S-Y1.csv' 
    v_test_file = 'data/V-Y1.csv'
    
    df_s = pd.read_csv(s_test_file, encoding='latin1').rename(columns=lambda x: 'time' if 'time' in x.lower() else x).sort_values('time')
    df_v = pd.read_csv(v_test_file, encoding='latin1').rename(columns=lambda x: 'time' if 'time' in x.lower() else x).sort_values('time')
    df_s['time'], df_v['time'] = df_s['time'].astype(float), df_v['time'].astype(float)
    df = pd.merge_asof(df_s, df_v, on='time', direction='nearest')
    
    raw_cols = [col for col in df.columns if ('acc' in col.lower() or 'gyr' in col.lower() or 'w' in col.lower()) and 'clean' not in col.lower()]
    for col in raw_cols: df[f'{col}_clean'] = apply_lowpass_filter(df[col])
    df = apply_virtual_alignment(df)
    
    with open('data/scaler_params.json', 'r') as f: scaler_params = json.load(f)
    mean_vals, scale_vals, feature_cols = np.array(scaler_params['mean']), np.array(scaler_params['scale']), scaler_params['features']
    
    speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()][0]
    dir_col = [col for col in df.columns if 'head' in col.lower() or 'yaw' in col.lower()][0]
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_cols + [speed_col, dir_col], inplace=True)
    
    X_scaled = (df[feature_cols].values - mean_vals) / scale_vals
    
    window_size = 100
    X_tensor = torch.tensor(np.array([X_scaled[i : i + window_size].T for i in range(len(X_scaled) - window_size)]), dtype=torch.float32)
    actual_speed = df[speed_col].values[window_size - 1 : -1]
    actual_dir = df[dir_col].values[window_size - 1 : -1]
    
    model = DualHeadVelocityCNN(num_features=len(feature_cols))
    model.load_state_dict(torch.load('data/bulk_velocity_cnn.pth', weights_only=True))
    model.eval()
    
    with torch.no_grad():
        ai_speed, ai_dir = model(X_tensor)
        ai_speed = ai_speed.numpy()
        ai_dir = ai_dir.numpy()
        
    # Create a 2-Panel Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    time_axis = np.arange(len(ai_speed)) / 10.0
    plot_limit = min(5000, len(ai_speed))
    
    # Plot 1: Speed
    ax1.plot(time_axis[:plot_limit], actual_speed[:plot_limit], color='black', linewidth=2.5, label='Actual Vehicle Speed')
    ax1.plot(time_axis[:plot_limit], ai_speed[:plot_limit], color='dodgerblue', linestyle='--', linewidth=1.5, label='AI Predicted Speed')
    ax1.set_title(f"Dual-Head Verification Test: Speed (Head 1)")
    ax1.set_ylabel("Speed")
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Direction / Heading
    ax2.plot(time_axis[:plot_limit], actual_dir[:plot_limit], color='black', linewidth=2.5, label='Actual Vehicle Heading')
    ax2.plot(time_axis[:plot_limit], ai_dir[:plot_limit], color='magenta', linestyle='--', linewidth=1.5, label='AI Predicted Heading')
    ax2.set_title(f"Dual-Head Verification Test: Direction (Head 2)")
    ax2.set_xlabel("Time (Seconds)")
    ax2.set_ylabel("Heading (Degrees)")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    out_img = 'data/dual_head_verification_plot.png'
    plt.savefig(out_img, dpi=300)
    print(f"Saved to {out_img}!")
    plt.show()