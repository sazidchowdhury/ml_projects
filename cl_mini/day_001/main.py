import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

# Load raw ECG from CSV (no header)
df = pd.read_csv("./ecg/Subject3F_ECG.csv", header=None)
ecg_signal = df.iloc[:, 0]

# Process ECG at 2000 Hz sampling frequency
signals, info = nk.ecg_process(ecg_signal, sampling_rate=2000)

# Visualize ECG with R‑peaks
nk.ecg_plot(signals, info)
plt.show()

# Extract R‑peaks indices
rpeaks = info["ECG_R_Peaks"]

# Compute HRV (RMSSD, SDNN) at 2000 Hz
hrv_time = nk.hrv_time(rpeaks, sampling_rate=2000, show=True)
plt.show()

# Print HRV results
print(f"RMSSD: {hrv_time['HRV_RMSSD'].values[0]:.2f} ms")
print(f"SDNN:  {hrv_time['HRV_SDNN'].values[0]:.2f} ms")
