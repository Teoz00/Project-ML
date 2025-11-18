# train_dqn.py
import gymnasium as gym
import argparse
import numpy as np
import torch
from dqn import DQNAgent
from utils import plot_learning_curve, set_seed, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="../results/dqn")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('Taxi-v3')
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = DQNAgent(n_states, n_actions, device=device, lr=1e-3, gamma=0.99,
                     buffer_size=20000, batch_size=64, update_every=1, epsilon_decay=30000)
    rewards = []
    losses = []
    ensure_dir(args.save_dir)
    step = 0
    for ep in range(1, args.episodes+1):
        s, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            a = agent.select_action(s)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.store(s, a, r, s2, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            s = s2
            ep_reward += r
            step += 1
        rewards.append(ep_reward)
        if ep % 100 == 0:
            print(f"[DQN] Ep {ep}/{args.episodes} last100_avg {np.mean(rewards[-100:]):.2f}")
    agent.save(args.save_dir)
    np.save(args.save_dir + "/rewards.npy", rewards)
    np.save(args.save_dir + "/losses.npy", np.array(losses))
    plot_learning_curve(rewards, filename=args.save_dir + "/learning_curve.png", title="DQN on Taxi-v3")
    env.close()

if __name__ == "__main__":
    main()
