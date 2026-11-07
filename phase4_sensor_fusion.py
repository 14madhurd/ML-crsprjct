import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from phase2_plot import VelocityCNN, StandardScaler

# ==========================================
# 1. LOAD DATA & GENERATE AI SPEED
# ==========================================
print("Loading data...")
df = pd.read_csv('data/phase1_processed.csv')

feature_cols = ['acc_x_aligned', 'acc_y_aligned', 'acc_z_aligned']
speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()][0]

X_raw = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

window_size = 20
X_windows = [X_scaled[i : i + window_size].T for i in range(len(X_scaled) - window_size)]
X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)

model = VelocityCNN(num_features=3)
model.load_state_dict(torch.load('data/velocity_cnn.pth', weights_only=True))
model.eval()

with torch.no_grad():
    ai_speed = model(X_tensor).numpy()

# Get ground truth for comparison and to simulate GNSS
actual_speed = df[speed_col].values[window_size - 1 : -1]

# ==========================================
# 2. SIMULATE GNSS & SIGNAL BLACKOUT
# ==========================================
print("Simulating GNSS outtages...")
dt = 0.1 # 10Hz
time_steps = len(ai_speed)

# Create a true position track
true_position = np.cumsum(actual_speed * dt)

# Simulate 1Hz GNSS with noise (GNSS is only available every 10th frame)
gnss_measurements = np.full(time_steps, np.nan)
gnss_noise_std = 2.0 # 2 meters of GPS noise
for i in range(0, time_steps, 10):
    gnss_measurements[i] = true_position[i] + np.random.normal(0, gnss_noise_std)

# SIMULATE A TUNNEL: Total GNSS blackout between seconds 60 and 150 (frames 600 to 1500)
blackout_start, blackout_end = 600, 1500
gnss_measurements[blackout_start:blackout_end] = np.nan 

# ==========================================
# 3. GNSS+INS FUSION ENGINE (Kalman Filter)
# ==========================================
print("Running Sensor Fusion Engine...")
# Initialize 1D Kalman Filter (State = [Position])
kf = KalmanFilter(dim_x=1, dim_z=1)
kf.x = np.array([[0.]])       # Initial position
kf.P = np.array([[10.]])      # Initial uncertainty
kf.H = np.array([[1.]])       # Measurement function
kf.R = np.array([[gnss_noise_std**2]]) # Measurement noise (GNSS inaccuracy)
kf.Q = np.array([[0.5]])      # Process noise (AI inaccuracy variance)

fused_positions = []

for i in range(time_steps):
    # STEP 1: PREDICT (Always runs at 10Hz using AI Speed)
    # distance = speed * time
    u = ai_speed[i] * dt 
    
    # State transition equation: New Pos = Old Pos + AI Distance
    kf.x = kf.x + u
    kf.P = kf.P + kf.Q
    
    # STEP 2: UPDATE (Deficit Handler Logic)
    # Only update if GNSS is available. If in a blackout (NaN), skip update!
    if not np.isnan(gnss_measurements[i]):
        kf.update(np.array([[gnss_measurements[i]]]))
        
    fused_positions.append(kf.x[0, 0])

# ==========================================
# 4. VISUALIZATION
# ==========================================
print("Generating Fusion Plot...")
time_axis = np.arange(time_steps) * dt
subset = 2500 # Plot first 250 seconds

plt.figure(figsize=(12, 6))

# Plot the GNSS readings (dots) - notice the massive gap where the tunnel is
plt.scatter(time_axis[:subset], gnss_measurements[:subset], color='orange', s=15, label='Noisy 1Hz GNSS', zorder=2)

# Plot the Ground Truth (black line)
plt.plot(time_axis[:subset], true_position[:subset], color='black', linewidth=2, label='True Trajectory', zorder=1)

# Plot the Kalman Filter Fusion (blue line)
plt.plot(time_axis[:subset], fused_positions[:subset], color='dodgerblue', linewidth=2.5, linestyle='--', label='Fused GNSS+INS Output', zorder=3)

# Highlight the Blackout Zone
plt.axvspan(blackout_start*dt, blackout_end*dt, color='red', alpha=0.15, label='GNSS Blackout (Tunnel)')

plt.title('Phase 4: Sensor Fusion with Deficit Handler (Seamless GNSS Outage Recovery)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Distance Traveled (Meters)')
plt.legend()
plt.grid(True)
plt.tight_layout()

output_img = 'data/sensor_fusion_plot.png'
plt.savefig(output_img, dpi=300)
print(f"Fusion plot saved to '{output_img}'!")
plt.show()