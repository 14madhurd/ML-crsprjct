import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import time

class IMUDataset(Dataset):
    def __init__(self, X_tensor, y_speed_tensor, y_dir_tensor):
        self.X = X_tensor
        self.y_speed = y_speed_tensor
        self.y_dir = y_dir_tensor
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_speed[idx], self.y_dir[idx]

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Hyper-Optimized Training on {device.type.upper()}\n")

    print("Loading bulk training data...")
    df = pd.read_csv('data/bulk_training_data.csv')
    
    acc_cols = ['acc_x_aligned', 'acc_y_aligned', 'acc_z_aligned']
    gyr_cols = [col for col in df.columns if ('gyr' in col.lower() or 'w' in col.lower()) and 'clean' in col.lower()]
    feature_cols = acc_cols + gyr_cols[:3] if len(gyr_cols) >= 3 else acc_cols
    
    speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()][0]
    dir_candidates = [col for col in df.columns if 'yaw' in col.lower() or 'steer' in col.lower() or 'head' in col.lower()]
    true_dir_cols = [col for col in dir_candidates if 'gyr' not in col.lower() and 'w' not in col.lower()]
    dir_col = true_dir_cols[0] if true_dir_cols else (gyr_cols[2] if len(gyr_cols) > 2 else gyr_cols[0])
        
    print("Sanitizing data...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_cols + [speed_col, dir_col], inplace=True)
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values)
    
    with open('data/scaler_params.json', 'w') as f:
        json.dump({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'features': feature_cols}, f)
        
    all_X_windows, all_y_speed, all_y_dir = [], [], []
    window_size = 100 
    
    grouped = df.groupby('dataset_id')
    print(f"Creating sliding windows across {len(grouped)} drives...")
    
    for dataset_id, group in grouped:
        X_group = group[feature_cols].values
        y_speed_group = group[speed_col].values
        y_dir_group = group[dir_col].values
        
        if len(X_group) <= window_size: continue
        for i in range(len(X_group) - window_size):
            all_X_windows.append(X_group[i : i + window_size].T)
            all_y_speed.append(y_speed_group[i + window_size - 1])
            all_y_dir.append(y_dir_group[i + window_size - 1])
            
    print("Splitting into 85% Train / 15% Validation...")
    X_train, X_val, y_speed_train, y_speed_val, y_dir_train, y_dir_val = train_test_split(
        all_X_windows, all_y_speed, all_y_dir, test_size=0.15, random_state=42
    )

    train_dataset = IMUDataset(
        torch.tensor(np.array(X_train), dtype=torch.float32), 
        torch.tensor(np.array(y_speed_train), dtype=torch.float32), 
        torch.tensor(np.array(y_dir_train), dtype=torch.float32)
    )
    val_dataset = IMUDataset(
        torch.tensor(np.array(X_val), dtype=torch.float32), 
        torch.tensor(np.array(y_speed_val), dtype=torch.float32), 
        torch.tensor(np.array(y_dir_val), dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    model = DualHeadVelocityCNN(num_features=len(feature_cols)).to(device)
    criterion_speed = nn.MSELoss()
    criterion_dir = nn.MSELoss()
    
    # Slightly lowered learning rate to prevent jumping past the perfect weights
    optimizer = optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5) 
    
    epochs = 50
    best_val_loss = float('inf')
    
    print(f"Starting Training with Best-Epoch Checkpointing...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_speed_loss, train_dir_loss, train_total_loss = 0.0, 0.0, 0.0
        
        for batch_X, batch_y_speed, batch_y_dir in train_loader:
            batch_X, batch_y_speed, batch_y_dir = batch_X.to(device), batch_y_speed.to(device), batch_y_dir.to(device)
            optimizer.zero_grad()
            pred_speed, pred_dir = model(batch_X)
            
            loss_speed = criterion_speed(pred_speed, batch_y_speed)
            loss_dir = criterion_dir(pred_dir, batch_y_dir)
            loss = loss_speed + (loss_dir * 10.0) 
            
            loss.backward()
            optimizer.step()
            
            train_speed_loss += loss_speed.item()
            train_dir_loss += loss_dir.item()
            train_total_loss += loss.item()
            
        avg_train_loss = train_total_loss / len(train_loader)
        
        # --- VALIDATION PHASE (NO LEARNING, JUST TESTING) ---
        model.eval()
        val_speed_loss, val_dir_loss, val_total_loss = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch_X, batch_y_speed, batch_y_dir in val_loader:
                batch_X, batch_y_speed, batch_y_dir = batch_X.to(device), batch_y_speed.to(device), batch_y_dir.to(device)
                pred_speed, pred_dir = model(batch_X)
                
                loss_speed = criterion_speed(pred_speed, batch_y_speed)
                loss_dir = criterion_dir(pred_dir, batch_y_dir)
                loss = loss_speed + (loss_dir * 10.0) 
                
                val_speed_loss += loss_speed.item()
                val_dir_loss += loss_dir.item()
                val_total_loss += loss.item()
                
        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_speed = val_speed_loss / len(val_loader)
        avg_val_dir = val_dir_loss / len(val_loader)
        
        scheduler.step(avg_val_loss) 
        epoch_time = time.time() - start_time
        
        # --- CHECKPOINT SAVING ---
        save_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'data/bulk_velocity_cnn.pth')
            save_msg = "⭐ NEW BEST MODEL SAVED!"
            
        print(f"Epoch {epoch+1}/{epochs} | Val Speed: {avg_val_speed:.2f} | Val Dir: {avg_val_dir:.2f} | {save_msg}")

    print("\nTraining Complete! The absolute best model has been locked into 'bulk_velocity_cnn.pth'.")