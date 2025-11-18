# double_dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import ReplayBuffer, ensure_dir
import time
from dqn import DQNNetwork

class DoubleDQNAgent:
    def __init__(self, n_states, n_actions, device='cpu', lr=1e-3, gamma=0.99,
                 buffer_size=20000, batch_size=64, target_update_every=1000,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=30000):
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.policy_net = DQNNetwork(n_states, n_actions).to(device)
        self.target_net = DQNNetwork(n_states, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)

        # epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0
        self.learn_step = 0
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

        q_values = self.policy_net(s).gather(1, a)

        with torch.no_grad():
            # action selection by policy_net
            a_prime = torch.argmax(self.policy_net(s2), dim=1, keepdim=True)
            # action evaluation by target_net
            q_target_next = self.target_net(s2).gather(1, a_prime)
            target = r + (1.0 - done) * self.gamma * q_target_next

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1
        # update target
        if self.total_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def save(self, path):
        ensure_dir(path)
        torch.save(self.policy_net.state_dict(), path + "/double_dqn.pt")
        torch.save(self.target_net.state_dict(), path + "/double_dqn_target.pt")
