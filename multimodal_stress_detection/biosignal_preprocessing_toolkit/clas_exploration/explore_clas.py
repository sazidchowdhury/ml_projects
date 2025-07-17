import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_BASE = '../../../../Datasets/CLAS/Participants/'
OUTPUT_BASE = 'plots_clas'
os.makedirs(OUTPUT_BASE, exist_ok=True)

FS = {'ecg': 256, 'eda': 128, 'ppg': 128}

def plot_main_session(csv_path: str, signal_type: str, fs: int, out_dir: str) -> None:
    df = pd.read_csv(csv_path)
    info_lines = [f"File: {csv_path}", f"Type: {signal_type}, FS: {fs} Hz"]

    for col in df.select_dtypes(include=[np.number]).columns:
        col_lower = col.lower()
        if signal_type == 'ecg' and 'ecg' not in col_lower: continue
        if signal_type == 'eda' and not any(x in col_lower for x in ['eda', 'gsr']): continue
        if signal_type == 'ppg' and 'ppg' not in col_lower: continue

        data = df[col].values
        if data.shape[0] < fs:
            continue

        duration = data.shape[0] / fs
        info_lines.append(f"{col}: {data.shape[0]} samples → {duration:.1f} s")
        n = min(data.shape[0], fs * 60)
        t = np.arange(n) / fs

        plt.figure(figsize=(10, 4))
        plt.plot(t, np.asarray(data[:n]))
        plt.title(f"{signal_type.upper()} ({col}) – first 60s")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plot_fname = f"{col}_60s.png"
        plot_path = os.path.join(out_dir, plot_fname)
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Plot saved: {plot_path}")

    info_path = os.path.join(out_dir, f"{Path(csv_path).stem}_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(info_lines))

def explore_clas_main() -> None:
    for part in os.listdir(DATA_BASE):
        sep = os.path.join(DATA_BASE, part, 'all_separate')
        if not os.path.isdir(sep):
            continue

        for fn in os.listdir(sep):
            fn_str = str(fn)
            if not fn_str.endswith('_all-png.csv'):
                continue

            lower = fn_str.lower()
            if 'ecg' in lower:
                typ = 'ecg'
            elif 'eda' in lower or 'gsr' in lower:
                typ = 'eda'
            elif 'ppg' in lower:
                typ = 'ppg'
            else:
                continue

            fs = FS[typ]
            out = os.path.join(OUTPUT_BASE, part, Path(fn_str).stem)
            os.makedirs(out, exist_ok=True)
            csv_path = os.path.join(sep, fn_str)

            print(f"→ Processing {part} | {fn_str} | type={typ}")
            plot_main_session(csv_path, typ, fs, out)

if __name__ == '__main__':
    explore_clas_main()
    print("🎉 CLAS main-session exploration complete.")
