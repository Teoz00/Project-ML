# src/train_q.py
import gymnasium as gym
import argparse
import numpy as np
import json
import os
from q_learning import TabularQLearning
from utils import plot_learning_curve, set_seed, ensure_dir, save_json, load_yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml")
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    c = cfg['q_learning']
    common = cfg['common']
    seed = args.seed if args.seed is not None else (common.get('seeds', [0])[0])
    set_seed(seed)

    env = gym.make(common['env_name'])
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    eps_decay_steps = int(c['episodes'] * c.get('epsilon_decay_frac', 0.8))
    agent = TabularQLearning(n_states, n_actions, alpha=c['alpha'], gamma=c['gamma'],
                             epsilon_start=c['epsilon_start'], epsilon_end=c['epsilon_end'],
                             epsilon_decay_steps=eps_decay_steps)

    save_dir = os.path.join(c['save_dir'], f"seed_{seed}")
    ensure_dir(save_dir)
    params = {'config': c, 'seed': seed}
    save_json(params, os.path.join(save_dir, "params.json"))

    rewards = agent.train(env, num_episodes=c['episodes'], max_steps_per_episode=c['max_steps'], eval_every=500, save_dir=save_dir)
    avg = agent.evaluate(env, episodes=100)
    print("Final evaluation avg reward (100 eps):", avg)
    np.save(os.path.join(save_dir, "rewards.npy"), rewards)
    plot_learning_curve(rewards, filename=os.path.join(save_dir, "learning_curve.png"), title="Q-Learning on Taxi-v3")
    env.close()

if __name__ == "__main__":
    main()
