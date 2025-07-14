import neurokit2 as nk
import matplotlib.pyplot as plt

# Load ECG data
data = nk.data("bio_eventrelated_100hz")
ecg_signal = data["ECG"] #type: ignore

# Process ECG: filter and detect R‑peaks
signals, info = nk.ecg_process(ecg_signal, sampling_rate=100)

# Plot ECG with R‑peaks marked
nk.ecg_plot(signals, info)
plt.show()

# Extract R‑peaks timestamps (sample indices)
rpeaks = info["ECG_R_Peaks"]

# Compute time-domain HRV metrics
hrv_time = nk.hrv_time(rpeaks, sampling_rate=100, show=True)

print(f"RMSSD: {hrv_time['HRV_RMSSD'].values[0]:.2f} ms")
print(f"SDNN:  {hrv_time['HRV_SDNN'].values[0]:.2f} ms")
