# src/results_aggregator.py
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from utils import ensure_dir

def gather(method_dir_pattern):
    # pattern: "results/dqn/seed_*"
    folders = sorted(glob.glob(method_dir_pattern))
    runs = []
    for f in folders:
        rpath = os.path.join(f, "rewards.npy")
        if os.path.exists(rpath):
            runs.append(np.load(rpath))
    return runs

def plot_compare(methods, save_path="results/comparison.png", window=50):
    plt.figure(figsize=(10,6))
    for method_name, pattern in methods.items():
        runs = gather(pattern)
        if not runs:
            continue
        min_len = min(len(r) for r in runs)
        arr = np.vstack([r[:min_len] for r in runs])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        # smooth mean with conv
        if len(mean) >= window:
            smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
            x = range(window-1, window-1+len(smooth))
            plt.plot(x, smooth, label=f"{method_name} mean")
            plt.fill_between(x, smooth-std[window-1:window-1+len(smooth)], smooth+std[window-1:window-1+len(smooth)], alpha=0.2)
        else:
            plt.plot(mean, label=f"{method_name} mean")
            plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Comparison mean ± std (multiple seeds)")
    plt.legend()
    plt.grid(True)
    ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()

if __name__ == "__main__":
    methods = {
        "Q-Learning": "results/q_learning/seed_*",
        "DQN": "results/dqn/seed_*",
        "DoubleDQN": "results/double_dqn/seed_*"
    }
    plot_compare(methods, save_path="results/comparison.png")
    print("Saved results/comparison.png")
