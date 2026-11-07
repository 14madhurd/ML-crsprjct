import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os

# ==========================================
# 1. DATA INGESTION & SYNCHRONIZATION
# ==========================================
def load_and_sync(s_path, v_path):
    print(f"Loading {s_path} and {v_path}...")
    
    df_s = pd.read_csv(s_path, encoding='latin1')
    df_v = pd.read_csv(v_path, encoding='latin1')
    
    # Identify the time columns (handling potential capitalization differences)
    time_col_s = [col for col in df_s.columns if 'time' in col.lower()][0]
    time_col_v = [col for col in df_v.columns if 'time' in col.lower()][0]
    
    # Standardize time column name to 'time' and sort
    df_s = df_s.sort_values(time_col_s).rename(columns={time_col_s: 'time'})
    df_v = df_v.sort_values(time_col_v).rename(columns={time_col_v: 'time'})
    
    # Force both time columns to be the exact same data type (float64)
    df_s['time'] = df_s['time'].astype(float)
    df_v['time'] = df_v['time'].astype(float)
    
    # Merge on the closest timestamp to align the 10Hz data perfectly
    df_synced = pd.merge_asof(df_s, df_v, on='time', direction='nearest')
    print(f"Successfully synced {len(df_synced)} rows.")
    return df_synced

# ==========================================
# 2. NOISE FILTRATION (Butterworth Low-Pass)
# ==========================================
def apply_lowpass_filter(data, cutoff_freq=2.0, sample_rate=10.0, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data.fillna(0))

# ==========================================
# 3. VIRTUAL ALIGNMENT (Rotation Matrix)
# ==========================================
def calculate_rotation_matrix(g_x, g_y, g_z):
    pitch = np.arctan2(-g_x, np.sqrt(g_y**2 + g_z**2))
    roll = np.arctan2(g_y, g_z)
    
    Ry = np.array([
        [np.cos(pitch),  0, np.sin(pitch)],
        [0,              1, 0            ],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0,             0            ],
        [0, np.cos(roll), -np.sin(roll) ],
        [0, np.sin(roll),  np.cos(roll) ]
    ])
    return np.dot(Ry, Rx)

def apply_virtual_alignment(df):
    # Find vehicle speed column (often 'speed', 'Velocity', or similar in CAN data)
    speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()]
    
    if not speed_col:
        print("Warning: Speed column not found. Using first 50 frames for stationary calibration.")
        stationary_data = df.head(50)
    else:
        stationary_data = df[df[speed_col[0]] < 0.5]
    
    if stationary_data.empty:
        stationary_data = df.head(50)
        
    # Get accelerometer column names dynamically
    acc_cols = [col for col in df.columns if 'acc' in col.lower() and 'clean' in col.lower()]
    
    if len(acc_cols) >= 3:
        g_x = stationary_data[acc_cols[0]].mean()
        g_y = stationary_data[acc_cols[1]].mean()
        g_z = stationary_data[acc_cols[2]].mean()
        
        R = calculate_rotation_matrix(g_x, g_y, g_z)
        acc_matrix = df[[acc_cols[0], acc_cols[1], acc_cols[2]]].values
        aligned_acc = np.dot(acc_matrix, R.T)
        
        df['acc_x_aligned'] = aligned_acc[:, 0]
        df['acc_y_aligned'] = aligned_acc[:, 1]
        df['acc_z_aligned'] = aligned_acc[:, 2]
        print("Virtual alignment applied successfully.")
    else:
        print("Warning: Could not find 3 cleaned acceleration axes for alignment.")
        
    return df

# ==========================================
# EXECUTE PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Starting Phase 1 pipeline...")
    
    # 1. Sync the exact files in your data folder
    df = load_and_sync('data/S-S1.csv', 'data/V-S1.csv')
    
    # 2. Filter accelerometer columns
    acc_raw_cols = [col for col in df.columns if 'acc' in col.lower() and 'clean' not in col.lower()]
    for col_name in acc_raw_cols:
        df[f'{col_name}_clean'] = apply_lowpass_filter(df[col_name])
            
    # 3. Align
    df_final = apply_virtual_alignment(df)
    
    # Export clean dataset
    output_path = 'data/phase1_processed.csv'
    df_final.to_csv(output_path, index=False)
    print(f"Phase 1 Complete. Processed data saved to '{output_path}'.")