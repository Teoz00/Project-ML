# train_double_dqn.py
import gymnasium as gym
import argparse
import numpy as np
import torch
from double_dqn import DoubleDQNAgent
from utils import plot_learning_curve, set_seed, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="../results/double_dqn")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('Taxi-v3')
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = DoubleDQNAgent(n_states, n_actions, device=device, lr=1e-3, gamma=0.99,
                           buffer_size=50000, batch_size=64, target_update_every=1000,
                           epsilon_decay=40000)
    rewards = []
    losses = []
    ensure_dir(args.save_dir)
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
        rewards.append(ep_reward)
        if ep % 100 == 0:
            print(f"[DoubleDQN] Ep {ep}/{args.episodes} last100_avg {np.mean(rewards[-100:]):.2f}")
    agent.save(args.save_dir)
    np.save(args.save_dir + "/rewards.npy", rewards)
    np.save(args.save_dir + "/losses.npy", np.array(losses))
    plot_learning_curve(rewards, filename=args.save_dir + "/learning_curve.png", title="Double DQN on Taxi-v3")
    env.close()

if __name__ == "__main__":
    main()
