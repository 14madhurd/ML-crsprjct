import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from phase2_plot import VelocityCNN, StandardScaler # Reusing your model architecture

# ==========================================
# 1. LOAD DATA, MODEL, & PREDICT
# ==========================================
print("Loading data and generating AI speeds...")
df = pd.read_csv('data/phase1_processed.csv')

# Ensure we have the Z-axis gyro for heading (Yaw)
gyr_z_col = [col for col in df.columns if 'gyr_z_aligned' in col or 'gyr_z' in col][0]

X_raw = df[['acc_x_aligned', 'acc_y_aligned', 'acc_z_aligned']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

window_size = 20
X_windows = [X_scaled[i : i + window_size].T for i in range(len(X_scaled) - window_size)]
X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)

model = VelocityCNN(num_features=3)
model.load_state_dict(torch.load('data/velocity_cnn.pth', weights_only=True))
model.eval()

with torch.no_grad():
    raw_ai_speed = model(X_tensor).numpy()

# ==========================================
# 2. SMOOTHING & KINEMATICS
# ==========================================
# Apply Savitzky-Golay filter to smooth the jittery AI speed
smoothed_ai_speed = savgol_filter(raw_ai_speed, window_length=15, polyorder=3)
# Prevent negative speeds (car doesn't reverse in this dataset slice)
smoothed_ai_speed = np.maximum(smoothed_ai_speed, 0)

# Extract aligned Z-axis gyroscope (yaw rate in radians/sec) matching the window
yaw_rate = df[gyr_z_col].values[window_size - 1 : -1]

# ==========================================
# 3. DEAD RECKONING 2D TRAJECTORY
# ==========================================
print("Calculating Dead Reckoning Path...")
dt = 0.1 # 10Hz sampling rate = 0.1 seconds per frame

x_pos, y_pos, heading = [0.0], [0.0], [0.0] # Starting at (0,0) facing East (0 rad)

for i in range(1, len(smoothed_ai_speed)):
    # Update heading: previous heading + (yaw_rate * dt)
    current_heading = heading[-1] + (yaw_rate[i] * dt)
    heading.append(current_heading)
    
    # Update X, Y position using Non-Holonomic Constraints (forward movement only)
    distance_step = smoothed_ai_speed[i] * dt
    x_pos.append(x_pos[-1] + distance_step * np.cos(current_heading))
    y_pos.append(y_pos[-1] + distance_step * np.sin(current_heading))

# ==========================================
# 4. PLOT THE TRAJECTORY
# ==========================================
plt.figure(figsize=(10, 10))
# Plot the first 60 seconds (600 frames) to see the initial driving path
subset = 600 

plt.plot(x_pos[:subset], y_pos[:subset], label='AI Dead Reckoning Path', color='crimson', linewidth=2.5)
plt.scatter(x_pos[0], y_pos[0], color='green', s=100, label='Start Point', zorder=5)
plt.scatter(x_pos[subset-1], y_pos[subset-1], color='red', s=100, label='End Point', zorder=5)

plt.title('Phase 3: GNSS-Denied Inertial Trajectory (IMU Only)')
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.legend()
plt.grid(True)
plt.axis('equal') # Ensure 1 meter X = 1 meter Y visually

output_img = 'data/trajectory_plot.png'
plt.savefig(output_img, dpi=300)
print(f"Trajectory plot saved to '{output_img}'!")
plt.show()