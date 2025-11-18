# src/evaluate.py
import gymnasium as gym
import argparse
import torch
import numpy as np
import os
from dqn import DQNAgent, DQNNetwork
from double_dqn import DoubleDQNAgent
from q_learning import TabularQLearning
from utils import load_yaml, set_seed

def evaluate_q(path, env, episodes=200):
    q_table = np.load(os.path.join(path, "q_table.npy"))
    agent = TabularQLearning(q_table.shape[0], q_table.shape[1])
    agent.Q = q_table
    return agent.evaluate(env, episodes=episodes)

def evaluate_dqn(path, env, device='cpu', episodes=200, double=False):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    if double:
        # load policy_net from double_dqn
        agent = DoubleDQNAgent(n_states, n_actions, device=device)
        agent.policy_net.load_state_dict(torch.load(os.path.join(path, "double_dqn.pt"), map_location=device))
        net = agent.policy_net
    else:
        agent = DQNAgent(n_states, n_actions, device=device)
        agent.policy_net.load_state_dict(torch.load(os.path.join(path, "dqn.pt"), map_location=device))
        net = agent.policy_net

    total = 0.0
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            s_t = torch.tensor([s], dtype=torch.long, device=device)
            with torch.no_grad():
                q = net(s_t)
                a = int(torch.argmax(q).item())
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
    return total / episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml")
    parser.add_argument('--path', type=str, required=True, help="path to model folder (results/.../seed_X)")
    parser.add_argument('--method', type=str, choices=['q','dqn','double'], required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(args.seed)
    env = gym.make(cfg['common']['env_name'])
    device = torch.device(cfg['common'].get('device','cpu'))

    if args.method == 'q':
        avg = evaluate_q(args.path, env, episodes=args.episodes)
    elif args.method == 'dqn':
        avg = evaluate_dqn(args.path, env, device=device, episodes=args.episodes, double=False)
    else:
        avg = evaluate_dqn(args.path, env, device=device, episodes=args.episodes, double=True)

    print(f"Average reward over {args.episodes} episodes: {avg:.2f}")

if __name__ == "__main__":
    main()
