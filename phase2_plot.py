import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. REBUILD THE EXACT ARCHITECTURE
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
# 2. LOAD DATA & MODEL
# ==========================================
print("Loading data and model for visualization...")
df = pd.read_csv('data/phase1_processed.csv')

feature_cols = ['acc_x_aligned', 'acc_y_aligned', 'acc_z_aligned']
speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()][0]

X_raw = df[feature_cols].values
y_raw = df[speed_col].values

# Re-apply the same scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Rebuild the sliding windows (2 seconds = 20 frames)
window_size = 20
X_windows = []
for i in range(len(X_scaled) - window_size):
    X_windows.append(X_scaled[i : i + window_size].T)

X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)

# Load the trained weights
model = VelocityCNN(num_features=len(feature_cols))
model.load_state_dict(torch.load('data/velocity_cnn.pth', weights_only=True))
model.eval() # Set to evaluation mode

# ==========================================
# 3. RUN INFERENCE & PLOT
# ==========================================
print("Running AI inference...")
with torch.no_grad():
    predictions = model(X_tensor).numpy()

# Align the actual speeds to the predictions
y_actual = y_raw[window_size - 1 : -1]

print("Generating plot...")
plt.figure(figsize=(12, 6))

# Plotting a 3000-frame subset (300 seconds / 5 mins of driving) 
# so the lines aren't too squished together to read
subset = 3000 
time_axis = np.arange(subset) / 10.0 # Convert 10Hz to seconds

plt.plot(time_axis, y_actual[:subset], label='Actual CAN Speed (Ground Truth)', color='black', linewidth=2)
plt.plot(time_axis, predictions[:subset], label='AI Predicted Speed (IMU only)', color='dodgerblue', alpha=0.8, linestyle='--')

plt.title('Intelligent Dead Reckoning: AI Velocity Prediction vs Ground Truth')
plt.xlabel('Time (Seconds)')
plt.ylabel('Speed')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot for your team to view
output_img = 'data/velocity_prediction_plot.png'
plt.savefig(output_img, dpi=300)
print(f"Plot saved to '{output_img}'!")

plt.show()