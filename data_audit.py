import numpy as np
import os

base_dir = "/Users/timele23/Library/Mobile Documents/com~apple~CloudDocs/Thesis_Master"

def audit_uea(name):
    base = os.path.join(base_dir, "data/raw", name)
    X_tr = np.load(os.path.join(base, "X_train.npy"))
    y_tr = np.load(os.path.join(base, "y_train.npy"))
    X_te = np.load(os.path.join(base, "X_test.npy"))
    y_te = np.load(os.path.join(base, "y_test.npy"))
    print("=== " + name + " ===")
    print("  X_train:", X_tr.shape, "  X_test:", X_te.shape)
    print("  Format: (samples, timesteps, channels)")
    classes = np.unique(y_tr).tolist()
    print("  Classes:", classes)
    if y_tr.dtype.kind in ('U', 'S', 'O'):
        from collections import Counter
        print("  Train class counts:", dict(Counter(y_tr.tolist())))
        print("  Test  class counts:", dict(Counter(y_te.tolist())))
    else:
        print("  Train class counts:", np.bincount(y_tr).tolist())
        print("  Test  class counts:", np.bincount(y_te).tolist())
    print("  X_train range: [{:.4f}, {:.4f}]  mean={:.4f}  std={:.4f}".format(
        float(X_tr.min()), float(X_tr.max()), float(X_tr.mean()), float(X_tr.std())))
    print("  X_test  range: [{:.4f}, {:.4f}]  mean={:.4f}  std={:.4f}".format(
        float(X_te.min()), float(X_te.max()), float(X_te.mean()), float(X_te.std())))
    print("  NaN train:", int(np.isnan(X_tr).sum()), "  NaN test:", int(np.isnan(X_te).sum()))
    print("  dtype:", X_tr.dtype)
    print()

audit_uea("BasicMotions")
audit_uea("Epilepsy")

print("=== DSA (processed) ===")
dsa_dir = os.path.join(base_dir, "data/processed/DSA")
if os.path.exists(dsa_dir):
    files = os.listdir(dsa_dir)
    print("  Processed files:", sorted(files))
    if "X.npy" in files:
        X = np.load(os.path.join(dsa_dir, "X.npy"))
        y = np.load(os.path.join(dsa_dir, "y.npy"))
        subjects = np.load(os.path.join(dsa_dir, "subjects.npy"))
        print("  X shape:", X.shape)
        print("  y shape:", y.shape, "  unique classes:", np.unique(y).tolist())
        print("  y counts:", np.bincount(y).tolist())
        print("  subjects unique:", np.unique(subjects).tolist())
        per_subj = {int(s): int((subjects==s).sum()) for s in np.unique(subjects)}
        print("  Samples per subject:", per_subj)
        print("  X range: [{:.4f}, {:.4f}]  mean={:.4f}  std={:.4f}".format(
            float(X.min()), float(X.max()), float(X.mean()), float(X.std())))
        print("  NaN:", int(np.isnan(X).sum()))
        print("  dtype:", X.dtype)
else:
    print("  NOT found at", dsa_dir)
print()

print("=== UCI raw check ===")
uci_dir = "/private/tmp/thesis_data/UCI"
if os.path.exists(uci_dir):
    subjects = sorted(os.listdir(uci_dir))
    print("  Subject folders: total", len(subjects), "->", subjects[:5])
    first_sub = os.path.join(uci_dir, subjects[0])
    if os.path.isdir(first_sub):
        files = sorted(os.listdir(first_sub))
        print("  Files in", subjects[0] + ": total", len(files), "->", files[:3])
        sample_file = os.path.join(first_sub, files[0])
        data = np.loadtxt(sample_file)
        print("  One segment shape:", data.shape, "  (timesteps x sensor_cols)")
        print("  Value range: [{:.4f}, {:.4f}]".format(float(data.min()), float(data.max())))
else:
    print("  Not found")
