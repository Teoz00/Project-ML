# dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import ReplayBuffer, ensure_dir
import time

class DQNNetwork(nn.Module):
    def __init__(self, n_states, n_actions, embedding_dim=64, hidden_dim=128):
        super().__init__()
        # use embedding for discrete state
        self.embed = nn.Embedding(n_states, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, s):
        # s: batch of ints
        x = self.embed(s).float()
        return self.net(x)

class DQNAgent:
    def __init__(self, n_states, n_actions, device='cpu', lr=1e-3, gamma=0.99,
                 buffer_size=10000, batch_size=64, update_every=1, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=20000):
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every

        self.policy_net = DQNNetwork(n_states, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)

        # epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learn_step = 0
        self.total_steps = 0

        self.loss_fn = nn.MSELoss()

    def epsilon(self):
        frac = min(1.0, self.total_steps / max(1, self.epsilon_decay))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state):
        eps = self.epsilon()
        self.total_steps += 1
        if np.random.rand() < eps:
            return np.random.randint(0, self.n_actions)
        s_t = torch.tensor([state], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = self.policy_net(s_t)
            return int(torch.argmax(q).item())

    def store(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        s = torch.tensor(batch.s, dtype=torch.long, device=self.device)
        a = torch.tensor(batch.a, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(batch.r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(batch.s2, dtype=torch.long, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(s).gather(1, a)  # Q(s,a)
        with torch.no_grad():
            q_next = self.policy_net(s2)  # Note: single net DQN (as requested)
            q_next_max, _ = torch.max(q_next, dim=1, keepdim=True)
            target = r + (1.0 - done) * self.gamma * q_next_max

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.learn_step += 1
        return loss.item()

    def save(self, path):
        ensure_dir(path)
        torch.save(self.policy_net.state_dict(), path + "/dqn.pt")
