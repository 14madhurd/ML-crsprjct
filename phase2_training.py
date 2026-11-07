import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SLIDING WINDOW DATASET
# ==========================================
class IMUDataset(Dataset):
    def __init__(self, features, targets, window_size=20):
        # window_size=20 means 2.0 seconds of data at 10Hz
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.X) - self.window_size
        
    def __getitem__(self, idx):
        # 1D-CNNs in PyTorch expect input shape: (Channels/Features, Sequence Length)
        x_window = self.X[idx : idx + self.window_size].T
        # Target is the vehicle speed at the very end of this 2-second window
        y_target = self.y[idx + self.window_size - 1] 
        return x_window, y_target

# ==========================================
# 2. 1D-CNN ARCHITECTURE
# ==========================================
class VelocityCNN(nn.Module):
    def __init__(self, num_features):
        super(VelocityCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Fully Connected Layer (Regression output)
        # Input size calculation: window_size(20) / pool1(2) / pool2(2) = 5
        self.fc1 = nn.Linear(64 * 5, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1) # Single output: Speed
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten for Dense layer
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze() # Remove extra dimensions

# ==========================================
# 3. TRAINING PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Loading Phase 1 data...")
    df = pd.read_csv('data/phase1_processed.csv')
    
    # Select features (Ensure these columns exist in your processed CSV)
    feature_cols = ['acc_x_aligned', 'acc_y_aligned', 'acc_z_aligned']
    # If you have gyroscope data, add it: ['gyr_x', 'gyr_y', 'gyr_z']
    
    # Find the speed column (Ground truth)
    speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()][0]
    
    X_raw = df[feature_cols].values
    y_raw = df[speed_col].values
    
    # Standardize features (Mean=0, Variance=1) - Crucial for Neural Networks
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Create DataLoader
    window_size = 20 # 2 seconds
    dataset = IMUDataset(X_scaled, y_raw, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize Model, Loss Function, and Optimizer
    model = VelocityCNN(num_features=len(feature_cols))
    criterion = nn.MSELoss() # Mean Squared Error for Regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 10
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Average MSE Loss: {epoch_loss/len(dataloader):.4f}")
        
    # Save the trained model for Phase 3/4
    torch.save(model.state_dict(), 'data/velocity_cnn.pth')
    print("Training complete! Model saved to 'data/velocity_cnn.pth'")