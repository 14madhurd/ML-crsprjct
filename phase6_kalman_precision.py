import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. 1D KALMAN FILTER FOR VELOCITY
# ==========================================
class VelocityKalmanFilter:
    def __init__(self, initial_state=0.0, process_noise=0.1, measurement_noise=2.0):
        self.state_estimate = initial_state      
        self.error_covariance = 1.0              
        self.Q = process_noise                   
        self.R = measurement_noise               
        
    def update(self, measurement):
        predicted_state = self.state_estimate
        predicted_error_cov = self.error_covariance + self.Q
        kalman_gain = predicted_error_cov / (predicted_error_cov + self.R)
        self.state_estimate = predicted_state + kalman_gain * (measurement - predicted_state)
        self.error_covariance = (1 - kalman_gain) * predicted_error_cov
        return self.state_estimate

# ==========================================
# 2. PREPROCESSING & CNN 
# ==========================================
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

class EnhancedVelocityCNN(nn.Module):
    def __init__(self, num_features):
        super(EnhancedVelocityCNN, self).__init__()
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
        self.fc1 = nn.Linear(256 * 12, 128) 
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        return self.fc2(torch.relu(self.drop(self.fc1(self.flatten(x))))).squeeze()

# ==========================================
# 3. PIPELINE & EVALUATION
# ==========================================
if __name__ == "__main__":
    # ---> REMEMBER TO SET THESE TO YOUR UNSEEN FILES <---
    s_unseen_file = 'data/S-Y1.csv' 
    v_unseen_file = 'data/V-Y1.csv'
    
    df_s = pd.read_csv(s_unseen_file, encoding='latin1').rename(columns=lambda x: 'time' if 'time' in x.lower() else x).sort_values('time')
    df_v = pd.read_csv(v_unseen_file, encoding='latin1').rename(columns=lambda x: 'time' if 'time' in x.lower() else x).sort_values('time')
    df_s['time'], df_v['time'] = df_s['time'].astype(float), df_v['time'].astype(float)
    df = pd.merge_asof(df_s, df_v, on='time', direction='nearest')
    
    raw_cols = [col for col in df.columns if ('acc' in col.lower() or 'gyr' in col.lower() or 'w' in col.lower()) and 'clean' not in col.lower()]
    for col in raw_cols: df[f'{col}_clean'] = apply_lowpass_filter(df[col])
    df = apply_virtual_alignment(df)
    
    with open('data/scaler_params.json', 'r') as f: scaler_params = json.load(f)
    mean_vals, scale_vals, feature_cols = np.array(scaler_params['mean']), np.array(scaler_params['scale']), scaler_params['features']
    speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()][0]
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_cols + [speed_col], inplace=True)
    
    X_scaled = (df[feature_cols].values - mean_vals) / scale_vals
    
    window_size = 100
    X_tensor = torch.tensor(np.array([X_scaled[i : i + window_size].T for i in range(len(X_scaled) - window_size)]), dtype=torch.float32)
    actual_speed = df[speed_col].values[window_size - 1 : -1]
    
    model = EnhancedVelocityCNN(num_features=len(feature_cols))
    model.load_state_dict(torch.load('data/bulk_velocity_cnn.pth', weights_only=True))
    model.eval()
    
    # This was the missing line that runs the model!
    with torch.no_grad(): ai_speed_raw = model(X_tensor).numpy()

    # --- APPLY RESPONSIVE KALMAN FILTER ---
    kf = VelocityKalmanFilter(initial_state=ai_speed_raw[0], process_noise=2.0, measurement_noise=0.5)
    ai_speed_smoothed = np.array([kf.update(measurement) for measurement in ai_speed_raw])
    
    # --- METRICS ---
    raw_rmse = np.sqrt(mean_squared_error(actual_speed, ai_speed_raw))
    smoothed_rmse = np.sqrt(mean_squared_error(actual_speed, ai_speed_smoothed))
    
    print(f"\n--- PRECISION METRICS ---")
    print(f"Raw AI RMSE:      {raw_rmse:.3f} m/s")
    print(f"Smoothed RMSE:    {smoothed_rmse:.3f} m/s")
    
    # --- DRIFT SIMULATION (40 SECOND OUTAGE) ---
    start_idx, end_idx = 1000, 1400
    if len(actual_speed) > end_idx:
        actual_distance = np.sum(actual_speed[start_idx:end_idx] * 0.1) 
        raw_distance = np.sum(ai_speed_raw[start_idx:end_idx] * 0.1)
        smoothed_distance = np.sum(ai_speed_smoothed[start_idx:end_idx] * 0.1)
        
        print(f"\n--- 40-SECOND OUTAGE DRIFT SIMULATION ---")
        print(f"Actual Distance Traveled: {actual_distance:.2f} meters")
        print(f"Raw AI Trajectory Error:  {abs(actual_distance - raw_distance):.2f} meters")
        print(f"Kalman Trajectory Error:  {abs(actual_distance - smoothed_distance):.2f} meters")
        
    plt.figure(figsize=(14, 7))
    time_axis = np.arange(len(ai_speed_raw)) / 10.0
    plot_limit = min(5000, len(ai_speed_raw))
    
    plt.plot(time_axis[:plot_limit], actual_speed[:plot_limit], color='black', linewidth=3.0, label='Actual CAN Speed')
    plt.plot(time_axis[:plot_limit], ai_speed_raw[:plot_limit], color='dodgerblue', alpha=0.4, linewidth=1.0, label='Raw AI Speed (Noisy)')
    plt.plot(time_axis[:plot_limit], ai_speed_smoothed[:plot_limit], color='magenta', linestyle='-', linewidth=2.0, label='Kalman Filtered AI (Precise)')
    
    plt.title(f"Kalman Precision Test on Unseen Holdout")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/kalman_precision_plot.png', dpi=300)
    plt.show()