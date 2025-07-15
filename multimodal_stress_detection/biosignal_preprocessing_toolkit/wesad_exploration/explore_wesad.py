import os, pickle

# 1. List subjects
folder = "../../../../Datasets/WESAD/"
subs = [f for f in os.listdir(folder) if f.startswith("S")]
print("Subjects:", subs)  # Expect 15 folders

# 2. Peek into a subject folder
sub = subs[0]
files = os.listdir(os.path.join(folder, sub))
print(f"{sub} files:", files)

# 3. Load data (pickle-based .pkl)
with open(os.path.join(folder, sub, f"{sub}.pkl"), "rb") as f:
    data = pickle.load(f, encoding="latin1")
print("Keys:", data.keys())  # e.g., 'signal', 'label', 'subject'