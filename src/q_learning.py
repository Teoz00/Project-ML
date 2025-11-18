# src/q_learning.py
import numpy as np
import time
from utils import ensure_dir, save_json
import os

class TabularQLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000):
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.step = 0

    def epsilon(self):
        frac = min(1.0, self.step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state):
        eps = self.epsilon()
        self.step += 1
        if np.random.rand() < eps:
            return np.random.randint(0, self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s2, done):
        target = r
        if not done:
            target += self.gamma * np.max(self.Q[s2])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def train(self, env, num_episodes=2000, max_steps_per_episode=200, eval_every=100, save_dir=None):
        rewards = []
        start = time.time()
        for ep in range(1, num_episodes+1):
            s, _ = env.reset()
            ep_reward = 0
            for t in range(max_steps_per_episode):
                a = self.select_action(s)
                s2, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                self.update(s, a, r, s2, done)
                s = s2
                ep_reward += r
                if done:
                    break
            rewards.append(ep_reward)
            if ep % eval_every == 0:
                print(f"[Q] Ep {ep}/{num_episodes} avg_reward(last{eval_every}): {np.mean(rewards[-eval_every:]):.2f}")
        duration = time.time() - start
        print(f"[Q] Training finished in {duration:.1f}s")
        if save_dir:
            ensure_dir(save_dir)
            np.save(os.path.join(save_dir, "q_table.npy"), self.Q)
            np.save(os.path.join(save_dir, "rewards.npy"), np.array(rewards))
        return rewards

    def evaluate(self, env, episodes=100, render=False):
        total = 0.0
        for _ in range(episodes):
            s, _ = env.reset()
            done = False
            while not done:
                a = int(np.argmax(self.Q[s]))
                s, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                total += r
        return total / episodes
