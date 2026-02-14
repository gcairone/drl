import argparse
import os
import time
from datetime import datetime
import numpy as np
import torch
from envs.humanoid_env import HumanoidEnv
from agents.ppo import PPOAgent
from utils.buffer import PPOBuffer
from utils.logger import Logger

def train(args):
    # generate unique run ID based on timestamp
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Starting training run: {run_id}")

    # logs will be saved in log/ with the run_id in the filename
    log_dir = os.path.join(args.log_path)
    
    # models will be saved in a subdirectory of save_dir named after the run_id
    model_save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(model_save_dir, exist_ok=True)

    logger = Logger(output_dir=log_dir, output_fname=f"{run_id}_metrics.csv")

    env = HumanoidEnv(render=False)
    obs_dim = env.obs_dim
    action_dim = env.action_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = PPOAgent(
        state_dim=obs_dim, 
        action_dim=action_dim, 
        lr=args.lr, 
        gamma=args.gamma, 
        clip_ratio=args.clip_ratio,
        train_pi_iters=args.pi_iters,
        train_v_iters=args.v_iters,
        target_kl=args.target_kl,
        device=str(device)
    )

    buffer = PPOBuffer(
        obs_dim=obs_dim, 
        act_dim=action_dim, 
        size=args.steps_per_epoch, 
        gamma=args.gamma, 
        lam=args.lam, 
        device=str(device)
    )

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = agent.load(args.resume)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"Resumed from epoch {start_epoch}")
            if 'buffer' in checkpoint:
                buffer.load_state_dict(checkpoint['buffer'])
                print("Buffer state restored.")
        else:
            print(f"Warning: Checkpoint file {args.resume} not found. Starting from scratch.")

    start_time = time.time()
    obs = env.reset()
    
    ep_ret, ep_len = 0, 0
    ep_returns = []
    ep_lengths = []

    for epoch in range(start_epoch, args.epochs):
        # 1. rollout phase
        for t in range(args.steps_per_epoch):
            
            action, info = agent.select_action(obs, training=True)
            val = info["value"]
            logp = info["log_prob"]
            next_obs, reward, done, _ = env.step(action)
            
            ep_ret += reward
            ep_len += 1

            buffer.store(obs, action, reward, val, logp)
            
            obs = next_obs

            timeout = ep_len == args.max_ep_len
            terminal = done or timeout
            epoch_ended = (t == args.steps_per_epoch - 1)

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    _, info_next = agent.select_action(obs, training=True)
                    last_val = info_next["value"]
                elif timeout:
                    _, info_next = agent.select_action(obs, training=True)
                    last_val = info_next["value"]
                else:
                    last_val = 0.0
                
                buffer.finish_path(last_val)

                if terminal:
                    ep_returns.append(ep_ret)
                    ep_lengths.append(ep_len)
                    
                    obs = env.reset()
                    ep_ret, ep_len = 0, 0

        # 2. pppdate phase
        metrics = agent.update(buffer)

        avg_ret = np.mean(ep_returns) if ep_returns else 0.0
        avg_len = np.mean(ep_lengths) if ep_lengths else 0.0
        ep_returns = [] # Reset for next epoch logging
        ep_lengths = []

        # prepare metrics dictionary
        epoch_metrics = {
            "Epoch": epoch + 1,
            "Ret": avg_ret,
            "Len": avg_len,
            "LossPi": metrics['loss_actor'],
            "LossV": metrics['loss_critic'],
            "Entropy": metrics['entropy'],
            "KL": metrics['kl'],
            "Time": time.time() - start_time
        }
        
        logger.log(epoch_metrics)

        print(f"Epoch: {epoch_metrics['Epoch']}/{args.epochs} | "
              f"Ret: {epoch_metrics['Ret']:.2f} | Len: {epoch_metrics['Len']:.1f} | "
              f"LossPi: {epoch_metrics['LossPi']:.4f} | "
              f"LossV: {epoch_metrics['LossV']:.4f} | "
              f"KL: {epoch_metrics['KL']:.4f} | "
              f"Time: {epoch_metrics['Time']:.0f}s")

        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth")
            agent.save(save_path, epoch=epoch+1, buffer=buffer.state_dict())
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_epoch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--pi_iters", type=int, default=80)
    parser.add_argument("--v_iters", type=int, default=80)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--save_dir", type=str, default="results/checkpoints")
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--log_path", type=str, default="results/logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    train(args)
