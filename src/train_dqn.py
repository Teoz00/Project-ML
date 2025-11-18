# src/train_dqn.py
import gymnasium as gym
import argparse
import numpy as np
import os
import torch
from dqn import DQNAgent
from utils import plot_learning_curve, set_seed, ensure_dir, save_json, load_yaml
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml")
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    c = cfg['dqn']
    common = cfg['common']
    seed = args.seed if args.seed is not None else (common.get('seeds', [0])[0])
    set_seed(seed)

    device = torch.device(common.get('device', 'cpu'))
    env = gym.make(common['env_name'])
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = DQNAgent(n_states, n_actions, device=device, lr=c['lr'], gamma=c['gamma'],
                     buffer_size=c['buffer_size'], batch_size=c['batch_size'],
                     epsilon_start=c['epsilon_start'], epsilon_end=c['epsilon_end'],
                     epsilon_decay=c['epsilon_decay_steps'],
                     embedding_dim=c.get('embedding_dim',64), hidden_dim=c.get('hidden_dim',128))

    save_dir = os.path.join(c['save_dir'], f"seed_{seed}")
    ensure_dir(save_dir)
    params = {'config': c, 'seed': seed}
    save_json(params, os.path.join(save_dir, "params.json"))

    tb_writer = None
    if c.get('tensorboard', False):
        tb_dir = os.path.join(c.get('tb_logdir','tb_logs/dqn'), f"seed_{seed}")
        tb_writer = SummaryWriter(tb_dir)

    rewards = []
    losses = []
    step = 0
    for ep in range(1, c['episodes']+1):
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
                if tb_writer:
                    tb_writer.add_scalar("loss", loss, step)
            s = s2
            ep_reward += r
            step += 1
            if step % 1000 == 0 and tb_writer:
                tb_writer.add_scalar("epsilon", agent.epsilon(), step)
        rewards.append(ep_reward)
        if ep % 100 == 0:
            last100 = np.mean(rewards[-100:]) if len(rewards)>=100 else np.mean(rewards)
            print(f"[DQN] Ep {ep}/{c['episodes']} last100_avg {last100:.2f}")
            if tb_writer:
                tb_writer.add_scalar("episode_reward", last100, ep)
    agent.save(save_dir)
    np.save(os.path.join(save_dir, "rewards.npy"), rewards)
    np.save(os.path.join(save_dir, "losses.npy"), np.array(losses))
    plot_learning_curve(rewards, filename=os.path.join(save_dir, "learning_curve.png"), title="DQN on Taxi-v3")
    if tb_writer:
        tb_writer.close()
    env.close()

if __name__ == "__main__":
    main()
