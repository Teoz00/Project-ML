# src/run_experiments.py
import os
import subprocess
import csv
import argparse
from utils import load_yaml, ensure_dir
import time

def run_cmd(cmd):
    print("RUN:", " ".join(cmd))
    return subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    common = cfg['common']
    seeds = common.get('seeds', [0])
    out_csv = cfg['run'].get('aggregate_save', "results/summary.csv")
    ensure_dir(os.path.dirname(out_csv) or ".")

    rows = []
    for method in ['q_learning', 'dqn', 'double_dqn']:
        for seed in seeds:
            start = time.time()
            if method == 'q_learning':
                cmd = ["python", "src/train_q.py", "--config", args.config, "--seed", str(seed)]
                res_dir = os.path.join(cfg['q_learning']['save_dir'], f"seed_{seed}")
            elif method == 'dqn':
                cmd = ["python", "src/train_dqn.py", "--config", args.config, "--seed", str(seed)]
                res_dir = os.path.join(cfg['dqn']['save_dir'], f"seed_{seed}")
            else:
                cmd = ["python", "src/train_double_dqn.py", "--config", args.config, "--seed", str(seed)]
                res_dir = os.path.join(cfg['double_dqn']['save_dir'], f"seed_{seed}")

            run_cmd(cmd)
            # read results
            rewards_path = os.path.join(res_dir, "rewards.npy")
            if os.path.exists(rewards_path):
                import numpy as np
                rewards = np.load(rewards_path)
                avg_final = float(rewards[-100:].mean()) if len(rewards) >= 100 else float(rewards.mean())
                std_final = float(rewards[-100:].std()) if len(rewards) >= 100 else float(rewards.std())
                duration = time.time() - start
                rows.append([method, seed, avg_final, std_final, len(rewards), round(duration,1), res_dir])
            else:
                rows.append([method, seed, "N/A", "N/A", 0, 0.0, res_dir])

    # save csv
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["method","seed","avg_reward_final","std_reward_final","episodes","duration_s","result_dir"])
        writer.writerows(rows)
    print("Saved summary to", out_csv)

if __name__ == "__main__":
    main()
