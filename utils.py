# utils.py
import random
import numpy as np
import torch
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import os

Transition = namedtuple('Transition', ('s', 'a', 'r', 's2', 'done'))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import gymnasium as gym
        gym.utils.seeding.np_random(seed)
    except Exception:
        pass

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

def plot_learning_curve(rewards, filename=None, title="Learning Curve", window=50):
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
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
