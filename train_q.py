# train_q.py
import gymnasium as gym
import argparse
import numpy as np
from q_learning import TabularQLearning
from utils import plot_learning_curve, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="../results/q_learning")
    args = parser.parse_args()

    set_seed(args.seed)
    env = gym.make('Taxi-v3')
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = TabularQLearning(n_states, n_actions, alpha=0.1, gamma=0.99,
                             epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=int(args.episodes*0.8))
    rewards = agent.train(env, num_episodes=args.episodes, max_steps_per_episode=200, eval_every=500, save_dir=args.save_dir)
    avg = agent.evaluate(env, episodes=100)
    print("Final evaluation avg reward (100 eps):", avg)
    plot_learning_curve(rewards, filename=args.save_dir + "/learning_curve.png", title="Q-Learning on Taxi-v3")
    np.save(args.save_dir + "/rewards.npy", rewards)
    env.close()

if __name__ == "__main__":
    main()
