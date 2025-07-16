import os, pickle
import numpy as np
import matplotlib.pyplot as plt

# Paths
data_dir = "../../../../Datasets/WESAD/"
output_dir = "../plots"
os.makedirs(output_dir, exist_ok=True)

# Sampling rates
fs_ecg = 700
fs_bvp = 64
fs_eda = 4
seg = 60  # seconds to plot


def process_subject(subject):
    pkl_path = os.path.join(data_dir, subject, f"{subject}.pkl")
    if not os.path.isfile(pkl_path):
        print(f"⚠️  Skipping {subject}: .pkl file not found.")
        return

    # Create subject-specific output folder
    subj_out = os.path.join(output_dir, subject)
    os.makedirs(subj_out, exist_ok=True)

    # Load pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    signals = data["signal"]
    chest = signals["chest"]
    wrist = signals["wrist"]
    labels = data.get("label", None)

    ecg = chest["ECG"]
    bvp = wrist["BVP"]
    eda = wrist["EDA"]

    # Time axes
    t_ecg = np.arange(len(ecg)) / fs_ecg
    t_bvp = np.arange(len(bvp)) / fs_bvp
    t_eda = np.arange(len(eda)) / fs_eda

    # Segments
    ecg_seg = ecg[: seg * fs_ecg]
    bvp_seg = bvp[: seg * fs_bvp]
    eda_seg = eda[: seg * fs_eda]

    t_ecg_seg = t_ecg[: seg * fs_ecg]
    t_bvp_seg = t_bvp[: seg * fs_bvp]
    t_eda_seg = t_eda[: seg * fs_eda]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_ecg_seg, ecg_seg)
    plt.title(f"{subject} – ECG (Chest, {fs_ecg} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.subplot(3, 1, 2)
    plt.plot(t_bvp_seg, bvp_seg, color="orange")
    plt.title(f"{subject} - BVP (Wrist, {fs_bvp} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.subplot(3, 1, 3)
    plt.plot(t_eda_seg, eda_seg, color="green")
    plt.title(f"{subject} - EDA (Wrist, {fs_eda} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plot_path = os.path.join(subj_out, f"{subject}_signals.png")
    plt.savefig(plot_path)
    plt.close()

    # Save basic info to text file
    info_txt = (
        f"ECG: {len(ecg)} samples → {len(ecg)/fs_ecg:.1f} s\n"
        f"BVP: {len(bvp)} samples → {len(bvp)/fs_bvp:.1f} s\n"
        f"EDA: {len(eda)} samples → {len(eda)/fs_eda:.1f} s\n"
        f"Labels present: {'Yes' if labels is not None else 'No'}\n"
    )
    
    info_path = os.path.join(subj_out, f"{subject}_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(info_txt)

    print(f"✅ Processed {subject}: plot saved to {plot_path}")


# Main loop
for sub in sorted(os.listdir(data_dir)):
    if sub.startswith("S") and len(sub) <= 3:  # e.g., S2, S10
        process_subject(sub)
