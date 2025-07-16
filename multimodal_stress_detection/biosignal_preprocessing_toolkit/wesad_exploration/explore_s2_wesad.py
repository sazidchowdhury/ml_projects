import os, pickle
import numpy as np
import matplotlib.pyplot as plt

# Paths
pkl_path = '../../../../Datasets/WESAD/S2/S2.pkl'
plot_dir = '../plots/wesad_plots/s2'
os.makedirs(plot_dir, exist_ok=True)

# 1) Load pickle with latin1 encoding to avoid Unicode errors
with open(pkl_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 2) Access nested dicts
signals = data['signal']
chest = signals['chest']
wrist = signals['wrist']
labels = data['label']  # shape ≈ (4,255,300,) at 700 Hz :contentReference[oaicite:1]{index=1}

# 3) Extract signals
fs_ecg = 700
fs_bvp = 64
fs_eda = 4

ecg = chest['ECG']
bvp = wrist['BVP']
eda = wrist['EDA']

# 4) Create time axes
t_ecg = np.arange(len(ecg)) / fs_ecg
t_bvp = np.arange(len(bvp)) / fs_bvp
t_eda = np.arange(len(eda)) / fs_eda

# 5) Select 60s segment
seg = 60
ecg_seg = ecg[:seg * fs_ecg]
bvp_seg = bvp[:seg * fs_bvp]
eda_seg = eda[:seg * fs_eda]

t_ecg_seg = t_ecg[:seg * fs_ecg]
t_bvp_seg = t_bvp[:seg * fs_bvp]
t_eda_seg = t_eda[:seg * fs_eda]

# 6) Plot
plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.plot(t_ecg_seg, ecg_seg); plt.title("ECG (Chest, 700 Hz) Segment")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.subplot(3,1,2)
plt.plot(t_bvp_seg, bvp_seg, color='orange'); plt.title("BVP (Wrist, 64 Hz)")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.subplot(3,1,3)
plt.plot(t_eda_seg, eda_seg, color='green'); plt.title("EDA (Wrist, 4 Hz)")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "S2_signals.png"))
plt.close()

# 7) Print info
print(f"ECG: {len(ecg)} samples → {len(ecg)/fs_ecg:.1f} s")
print(f"BVP: {len(bvp)} samples → {len(bvp)/fs_bvp:.1f} s")
print(f"EDA: {len(eda)} samples → {len(eda)/fs_eda:.1f} s")
print(f"Labels length: {len(labels)}")
