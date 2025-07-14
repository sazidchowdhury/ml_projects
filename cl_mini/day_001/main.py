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
