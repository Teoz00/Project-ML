# src/utils.py
import os
import json
import random
import yaml
import time
import numpy as np
import torch
from collections import deque, namedtuple
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', ('s', 'a', 'r', 's2', 'done'))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import gymnasium as gym
        # gym seeding best-effort
        try:
            gym.utils.seeding.np_random(seed)
        except Exception:
            pass
    except Exception:
        pass

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append(Transition(s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

def plot_learning_curve(rewards, filename=None, title="Learning Curve", window=50, show_std=False, runs=None):
    plt.figure(figsize=(8,5))
    rewards = np.array(rewards)
    plt.plot(rewards, label='episode reward')
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, window-1+len(smooth)), smooth, label=f'{window}-ep moving avg')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if filename:
        ensure_dir(os.path.dirname(filename) or ".")
        plt.savefig(filename, bbox_inches='tight', dpi=200)
    else:
        plt.show()
    plt.close()

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())
