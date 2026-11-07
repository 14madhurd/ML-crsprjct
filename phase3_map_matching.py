import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import osmnx as ox
from scipy.signal import savgol_filter
from shapely.geometry import Point
from phase2_plot import VelocityCNN, StandardScaler

# ==========================================
# 1. GENERATE DEAD RECKONING TRAJECTORY
# ==========================================
print("Loading data and generating AI trajectory...")
df = pd.read_csv('data/phase1_processed.csv')

# Use aligned IMU data
feature_cols = ['acc_x_aligned', 'acc_y_aligned', 'acc_z_aligned']

# More robust search for the Z-axis gyroscope column
gyr_z_candidates = [col for col in df.columns if ('gyr' in col.lower() or 'w' in col.lower()) and 'z' in col.lower()]

X_raw = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

window_size = 20
X_windows = [X_scaled[i : i + window_size].T for i in range(len(X_scaled) - window_size)]
X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)

# Load Model
model = VelocityCNN(num_features=3)
model.load_state_dict(torch.load('data/velocity_cnn.pth', weights_only=True))
model.eval()

with torch.no_grad():
    raw_ai_speed = model(X_tensor).numpy()

# Smooth speed and safely extract yaw
smoothed_ai_speed = np.maximum(savgol_filter(raw_ai_speed, window_length=15, polyorder=3), 0)

if len(gyr_z_candidates) > 0:
    gyr_z_col = gyr_z_candidates[0]
    print(f"Found Gyroscope Z column: '{gyr_z_col}'")
    yaw_rate = df[gyr_z_col].values[window_size - 1 : -1]
else:
    print(f"Warning: No gyroscope column found. Available columns: {df.columns.tolist()}")
    print("Simulating a straight path (0 yaw rate).")
    yaw_rate = np.zeros(len(smoothed_ai_speed))
# Calculate 2D relative path
dt = 0.1
x_pos, y_pos, heading = [0.0], [0.0], [0.0]
for i in range(1, len(smoothed_ai_speed)):
    current_heading = heading[-1] + (yaw_rate[i] * dt)
    heading.append(current_heading)
    x_pos.append(x_pos[-1] + (smoothed_ai_speed[i] * dt) * np.cos(current_heading))
    y_pos.append(y_pos[-1] + (smoothed_ai_speed[i] * dt) * np.sin(current_heading))

# ==========================================
# 2. OPENSTREETMAP MAP-MATCHING
# ==========================================
print("Downloading OpenStreetMap road network...")
# We use a proxy start location (e.g., a sample coordinate from the IO-VNBD dataset area)
# You can replace this with the first valid GPS coordinate before the "tunnel/outage"
start_lat, start_lon = 39.9847, 116.3184 # Example: Beijing (common in VNBD datasets)

# Download the drivable road network within a 1km radius
graph = ox.graph_from_point((start_lat, start_lon), dist=1000, network_type='drive')
nodes, edges = ox.graph_to_gdfs(graph)

print("Applying Map-Matching constraints...")
# Convert our relative X/Y meters to approximate Latitude/Longitude
# 1 degree of latitude is approx 111,000 meters
lat_path = start_lat + (np.array(y_pos) / 111000.0)
lon_path = start_lon + (np.array(x_pos) / (111000.0 * np.cos(np.radians(start_lat))))

snapped_lats, snapped_lons = [], []

# Snap every 10th point (1 second intervals) to the nearest road edge to prevent drift
subset = 1500 # Simulate 150 seconds of GPS-denied driving
for i in range(0, subset, 10): 
    point = (lat_path[i], lon_path[i])
    # Find the nearest physical road
    nearest_edge = ox.distance.nearest_edges(graph, point[1], point[0])
    
    # Get the geometry of that road segment
    edge_geom = edges.loc[nearest_edge]['geometry']
    if isinstance(edge_geom, pd.Series):
        edge_geom = edge_geom.iloc[0]
        
    # Snap our drifting point precisely onto the road line
    snapped_point = edge_geom.interpolate(edge_geom.project(Point(point[1], point[0])))
    snapped_lons.append(snapped_point.x)
    snapped_lats.append(snapped_point.y)

# ==========================================
# 3. VISUALIZATION
# ==========================================
print("Generating Map-Matched trajectory plot...")
fig, ax = ox.plot_graph(graph, show=False, close=False, edge_color='#999999', edge_alpha=0.5, node_size=0)

# Plot the raw, drifting Dead Reckoning path
ax.plot(lon_path[:subset], lat_path[:subset], color='red', linestyle='--', linewidth=2, label='Drifting IMU Path')

# Plot the Snapped, Map-Matched path
ax.plot(snapped_lons, snapped_lats, color='cyan', linewidth=3, label='Snapped Path (Map-Matched)')
ax.scatter(start_lon, start_lat, c='lime', s=100, label='Start Point', zorder=5)

plt.title('Phase 3: Map-Matching (Snapping to OSM Road Network)')
plt.legend()
plt.tight_layout()

output_img = 'data/map_matched_plot.png'
plt.savefig(output_img, dpi=300, facecolor='black')
print(f"Map-matching plot saved to '{output_img}'!")
plt.show()