import argparse
import os
import time
from datetime import datetime
import numpy as np
import torch
from envs.humanoid_env import RANDOMIZATION_DEFAULTS
from envs.vec_env import VecHumanoidEnv
from agents.ppo import PPOAgent
from utils.buffer import PPOBuffer
from utils.curriculum import DifficultyCurriculum
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

    if args.seed is not None:
        torch.manual_seed(args.seed)

    assert args.steps_per_epoch % args.num_envs == 0, "steps_per_epoch must be divisible by num_envs"
    steps_per_env = args.steps_per_epoch // args.num_envs

    env_kwargs = dict(noise_scale=args.noise_scale, min_height=args.min_height, term_height=args.term_height, frame_skip=args.frame_skip)
    env_kwargs.update({k: getattr(args, k) for k in RANDOMIZATION_DEFAULTS})
    envs = VecHumanoidEnv(args.num_envs, seed=args.seed, render=False, **env_kwargs)
    obs_dim = envs.obs_dim
    action_dim = envs.action_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = PPOAgent(
        state_dim=obs_dim, 
        action_dim=action_dim, 
        lr=args.lr, 
        gamma=args.gamma, 
        clip_ratio=args.clip_ratio,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        device=str(device),
        obs_norm=args.obs_norm,
        log_std_min=args.log_std_min if np.isfinite(args.log_std_min) else None
    )

    # randomization level: adaptive curriculum, or a fixed level with --no-curriculum
    curriculum = DifficultyCurriculum(start=args.difficulty, step=args.curr_step, up=args.curr_up,
                                      down=args.curr_down, min_episodes=args.curr_min_episodes)

    buffer = PPOBuffer(
        obs_dim=obs_dim, 
        act_dim=action_dim, 
        size=args.steps_per_epoch, 
        gamma=args.gamma, 
        lam=args.lam, 
        device=str(device),
        num_envs=args.num_envs
    )

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = agent.load(args.resume)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"Resumed from epoch {start_epoch}")
            if args.curriculum and 'curriculum' in checkpoint:
                curriculum.load_state_dict(checkpoint['curriculum'])
                print(f"Curriculum restored at difficulty {curriculum.level:.2f}.")
            if 'buffer' in checkpoint:
                if np.shape(checkpoint['buffer']['obs_buf']) == buffer.obs_buf.shape:
                    buffer.load_state_dict(checkpoint['buffer'])
                    print("Buffer state restored.")
                else:
                    print("Buffer state skipped (saved with a different num_envs/steps_per_epoch).")
        else:
            print(f"Warning: Checkpoint file {args.resume} not found. Starting from scratch.")

    print(f"Randomization difficulty: {curriculum.level:.2f} ({'adaptive curriculum' if args.curriculum else 'fixed'})")
    envs.set_difficulty(curriculum.level)

    start_time = time.time()
    obs = envs.reset()
    
    ep_ret = np.zeros(args.num_envs)
    ep_len = np.zeros(args.num_envs, dtype=np.int64)
    ep_returns = []
    ep_lengths = []
    ep_success = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # 1. rollout phase: all envs step together, one batched forward pass per step
        for t in range(steps_per_env):
            
            action, info = agent.select_action(obs, training=True)
            next_obs, reward, done, _ = envs.step(action)
            
            ep_ret += reward
            ep_len += 1

            buffer.store(obs, action, reward, info["value"], info["log_prob"])
            
            obs = next_obs

            timeout = ep_len == args.max_ep_len
            epoch_ended = (t == steps_per_env - 1)
            ended = done | timeout | epoch_ended
            if not ended.any():
                continue

            # bootstrap with V(s_T) for trajectories cut by timeout or epoch end, 0 for terminal states
            if (ended & ~done).any():
                _, info_next = agent.select_action(obs, training=True)
                last_vals = info_next["value"]

            for i in np.flatnonzero(ended):
                buffer.finish_path(0.0 if done[i] else last_vals[i], env_idx=i)

                if done[i] or timeout[i]:
                    ep_returns.append(ep_ret[i])
                    ep_lengths.append(ep_len[i])
                    ep_success.append(not done[i])  # reached the time limit without falling
                    curriculum.record(not done[i])
                    
                    obs[i] = envs.reset_at(i)
                    ep_ret[i], ep_len[i] = 0.0, 0

        rollout_time = time.time() - epoch_start

        # 2. update phase
        metrics = agent.update(buffer)
        update_time = time.time() - epoch_start - rollout_time

        avg_ret = np.mean(ep_returns) if ep_returns else 0.0
        avg_len = np.mean(ep_lengths) if ep_lengths else 0.0
        n_episodes = len(ep_returns)  # 0 when no episode finished in this epoch (Ret and Len then read 0)
        success_rate = float(np.mean(ep_success)) if ep_success else float("nan")
        ep_returns = [] # Reset for next epoch logging
        ep_lengths = []
        ep_success = []

        # difficulty used during this epoch is logged; a change applies from the next episode resets on
        difficulty = curriculum.level
        if args.curriculum and curriculum.update():
            envs.set_difficulty(curriculum.level)

        # prepare metrics dictionary
        epoch_metrics = {
            "Epoch": epoch + 1,
            "Ret": avg_ret,
            "Len": avg_len,
            "LenSec": avg_len * envs.envs[0].dt,
            "LossPi": metrics['loss_actor'],
            "LossV": metrics['loss_critic'],
            "Entropy": metrics['entropy'],
            "KL": metrics['kl'],
            "UpdEpochs": metrics['update_epochs'],
            "Eps": n_episodes,
            "Succ": success_rate,
            "Diff": difficulty,
            "Time": time.time() - start_time,
            "SPS": args.steps_per_epoch / (rollout_time + update_time),
            "TRollout": rollout_time,
            "TUpdate": update_time
        }
        
        logger.log(epoch_metrics)

        print(f"Epoch: {epoch_metrics['Epoch']}/{args.epochs} | "
              f"Ret: {epoch_metrics['Ret']:.2f} | Len: {epoch_metrics['Len']:.1f} ({epoch_metrics['LenSec']:.2f}s) | "
              f"LossPi: {epoch_metrics['LossPi']:.4f} | "
              f"LossV: {epoch_metrics['LossV']:.4f} | "
              f"KL: {epoch_metrics['KL']:.4f} | "
              f"SPS: {epoch_metrics['SPS']:.0f} | "
              f"Time: {epoch_metrics['Time']:.0f}s | "
              f"Diff: {epoch_metrics['Diff']:.2f} | Succ: {epoch_metrics['Succ']:.2f}")

        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth")
            agent.save(save_path, epoch=epoch+1, buffer=buffer.state_dict(), env_kwargs=env_kwargs, curriculum=curriculum.state_dict())
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_epoch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--update_epochs", type=int, default=10, help="Passes over the batch per update")
    parser.add_argument("--minibatch_size", type=int, default=64, help="SGD minibatch size (<= 0: full batch)")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Gradient clipping (<= 0: off)")
    parser.add_argument("--target_kl", type=float, default=0.01, help="Early stop at KL > 1.5 * target_kl (<= 0: off)")
    parser.add_argument("--save_dir", type=str, default="results/checkpoints")
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--log_path", type=str, default="results/logs")
    parser.add_argument("--num_envs", type=int, default=8, help="Parallel environments (batched inference)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for torch and the environments")
    parser.add_argument("--noise_scale", type=float, default=0.01, help="Uniform noise added to qpos/qvel at reset")
    parser.add_argument("--min_height", type=float, default=1.0, help="Torso height above which the alive reward is given")
    parser.add_argument("--term_height", type=float, default=1.0, help="Torso height below which the episode terminates")
    parser.add_argument("--frame_skip", type=int, default=3, help="Physics steps per action")
    parser.add_argument("--obs_norm", action=argparse.BooleanOptionalAction, default=True, help="Normalize observations with running mean/std")
    parser.add_argument("--log_std_min", type=float, default=-1.2,
                        help="Floor of the actor's log standard deviation (-inf: no floor)")
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True,
                        help="Adapt the randomization difficulty to the success rate (--no-curriculum: keep --difficulty fixed)")
    parser.add_argument("--difficulty", type=float, default=0.0,
                        help="Randomization level in [0, 1]: the starting level of the curriculum, or the fixed level without it")
    parser.add_argument("--curr_step", type=float, default=0.02, help="Curriculum: level change per decision")
    parser.add_argument("--curr_up", type=float, default=0.7, help="Curriculum: raise the level at this success rate or above")
    parser.add_argument("--curr_down", type=float, default=0.3, help="Curriculum: lower the level below this success rate")
    parser.add_argument("--curr_min_episodes", type=int, default=20, help="Curriculum: finished episodes needed for each decision")
    for name, default in RANDOMIZATION_DEFAULTS.items():
        parser.add_argument(f"--{name}", type=float, default=default, help=f"Randomization amplitude at difficulty 1 (default {default})")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    train(args)
