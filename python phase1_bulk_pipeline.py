import os
import glob
import pandas as pd
import numpy as np
import warnings
from scipy.signal import butter, filtfilt

# Ignore pandas fragmentation warnings during bulk merge
warnings.filterwarnings('ignore')

# ==========================================
# 1. CORE FUNCTIONS 
# ==========================================
def load_and_sync(s_path, v_path):
    df_s = pd.read_csv(s_path, encoding='latin1')
    df_v = pd.read_csv(v_path, encoding='latin1')
    
    time_col_s = [col for col in df_s.columns if 'time' in col.lower()][0]
    time_col_v = [col for col in df_v.columns if 'time' in col.lower()][0]
    
    df_s = df_s.sort_values(time_col_s).rename(columns={time_col_s: 'time'})
    df_v = df_v.sort_values(time_col_v).rename(columns={time_col_v: 'time'})
    
    df_s['time'] = df_s['time'].astype(float)
    df_v['time'] = df_v['time'].astype(float)
    
    return pd.merge_asof(df_s, df_v, on='time', direction='nearest')

def apply_lowpass_filter(data, cutoff_freq=2.0, sample_rate=10.0, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data.fillna(0))

def calculate_rotation_matrix(g_x, g_y, g_z):
    pitch = np.arctan2(-g_x, np.sqrt(g_y**2 + g_z**2))
    roll = np.arctan2(g_y, g_z)
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    return np.dot(Ry, Rx)

def apply_virtual_alignment(df):
    speed_col = [col for col in df.columns if 'speed' in col.lower() or 'vel' in col.lower()]
    stationary_data = df[df[speed_col[0]] < 0.5] if speed_col else df.head(50)
    if stationary_data.empty: stationary_data = df.head(50)
        
    acc_cols = [col for col in df.columns if 'acc' in col.lower() and 'clean' in col.lower()]
    if len(acc_cols) >= 3:
        g_x, g_y, g_z = stationary_data[acc_cols[0]].mean(), stationary_data[acc_cols[1]].mean(), stationary_data[acc_cols[2]].mean()
        R = calculate_rotation_matrix(g_x, g_y, g_z)
        acc_matrix = df[[acc_cols[0], acc_cols[1], acc_cols[2]]].values
        aligned_acc = np.dot(acc_matrix, R.T)
        df['acc_x_aligned'], df['acc_y_aligned'], df['acc_z_aligned'] = aligned_acc[:, 0], aligned_acc[:, 1], aligned_acc[:, 2]
    return df

# ==========================================
# 2. BULK EXECUTION ENGINE
# ==========================================
if __name__ == "__main__":
    print("Starting Bulk Processing Pipeline...")
    
    # Find all Smartphone files in the data folder
    s_files = glob.glob('data/S-*.csv')
    processed_datasets = []
    
    print(f"Found {len(s_files)} Smartphone datasets. Looking for Vehicle matches...")
    
    for s_path in s_files:
        identifier = os.path.basename(s_path).replace('S-', '').replace('.csv', '')
        v_path = f"data/V-{identifier}.csv"
        
        if os.path.exists(v_path):
            print(f"[{identifier}] Processing pair...")
            try:
                # 1. Sync Data
                df = load_and_sync(s_path, v_path)
                
                # 2. Clean Accelerometer and Gyroscope Noise
                raw_cols = [col for col in df.columns if ('acc' in col.lower() or 'gyr' in col.lower() or 'w' in col.lower()) and 'clean' not in col.lower()]
                for col in raw_cols:
                    df[f'{col}_clean'] = apply_lowpass_filter(df[col])
                
                # 3. Align Virtual Gravity
                df_final = apply_virtual_alignment(df)
                
                # Tag with dataset ID so Phase 2 knows how to group the sliding windows
                df_final['dataset_id'] = identifier
                processed_datasets.append(df_final)
                
            except Exception as e:
                print(f"[{identifier}] Error processing: {e}")
        else:
            print(f"Warning: No matching Vehicle file found for {s_path}")

    # ==========================================
    # 3. CONCATENATE & EXPORT
    # ==========================================
    if processed_datasets:
        print(f"\nSuccessfully processed {len(processed_datasets)} pairs. Stitching together...")
        master_df = pd.concat(processed_datasets, ignore_index=True)
        
        output_path = 'data/bulk_training_data.csv'
        master_df.to_csv(output_path, index=False)
        print(f"Total Rows Extracted: {len(master_df)}")
        print(f"Bulk data exported successfully to '{output_path}'!")
    else:
        print("No datasets were processed successfully.")