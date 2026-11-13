# XLR8 Dead Reckoning - Android Edge Application

An edge-native Android application that integrates **Intelligent Dead Reckoning (IDR)** powered by the PyTorch Dual-Head 1D-CNN model and an OpenStreetMap real-time map with automatic GNSS outage detection.

---

## Architecture & Features

1. **Embedded Neural Engine**:
   - Packaged with `bulk_velocity_cnn.pt` (TorchScript format) and `scaler_params.json` in `app/src/main/assets/`.
   - Runs on-device inference via PyTorch Mobile with low latency (<5ms per 10Hz inference).
   - Dynamic virtual alignment using stationary gravity decomposition to cancel phone mount tilt.
   - 4th-order zero-phase Butterworth low-pass filtering at 2.0 Hz to eliminate vehicle vibrations.

2. **Real-Time OpenStreetMap Integration**:
   - Powered by `osmdroid` connecting over the internet to live OpenStreetMap tile servers.
   - Zero API keys required.
   - Vehicle heading orientation arrow with dynamic color feedback:
     - Blue arrow: GNSS satellite navigation.
     - Amber arrow: Dead Reckoning (during GNSS outage).
   - Real-time polyline trajectory tracing.

3. **Automatic GNSS Outage Detection & Deficit Recovery**:
   - Watchdog timer monitors satellite fix intervals.
   - If GNSS updates cease for > 2.0 seconds (e.g. entering a tunnel, underground parking, or urban canyon), the system automatically transitions into Dead Reckoning mode.
   - Propagates latitude, longitude, and heading at 10Hz using AI-inferred speed and gyroscope yaw integration.
   - When GNSS fixes return (e.g. exiting a tunnel), the state seamlessly resynchronizes.
   - Includes a "Simulate GNSS Outage" switch to test the entire failover flow anywhere on-demand.

4. **Clean, Minimalist UI**:
   - Modern slate theme without emojis.
   - Live telemetry grid: Speed (km/h & m/s), Heading & Cardinal bearing, Outage Duration, Dead Reckoning Distance, and Accuracy.

---

## How to Open and Run in Android Studio

1. **Open in Android Studio**:
   - Launch **Android Studio**.
   - Select **File -> Open...** and browse to:
     `c:\Users\ASUS\Desktop\ML\android_app`
   - Click **OK**.

2. **Gradle Sync**:
   - Android Studio will automatically detect the Gradle files and download the required libraries (`osmdroid`, `pytorch_android`, `play-services-location`).
   - If prompted, accept the Android SDK Platform 34 installation.

3. **Run on Device or Emulator**:
   - Connect an Android device (with USB debugging enabled) or start an Android Virtual Device (AVD).
   - Press **Run 'app'** (`Shift + F10` or the green Play icon).
   - Grant the Location permission when prompted.

4. **Using the App**:
   - Tap **Start Navigation** to begin 10Hz IMU stream and GNSS tracking.
   - Tap **Recalibrate** while holding the vehicle stationary for 5 seconds to calibrate phone orientation.
   - Toggle **Simulate GNSS Outage (Tunnel)** to see Dead Reckoning automatically take over navigation.
