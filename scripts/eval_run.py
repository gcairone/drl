"""
Evaluation of running checkpoints under a fixed command and fixed perturbations.

    python3 -m scripts.eval_run --model <ckpt> --speeds 1 2 3 --perturbation 0 0.5

Every row runs `--episodes` independent episodes with the same commanded speed and heading offset and
reports how well the humanoid follows it.
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.loading import load_agent
from envs.run_env import HumanoidRunEnv


def run_command(agent, env_kwargs: Dict, speed: float, heading: float, perturbation: float,
                stochastic: bool, episodes: int, max_steps: int, seed: int,
                assist: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Runs `episodes` episodes in lockstep (one batched forward pass per step over the envs still alive)."""
    envs = [HumanoidRunEnv(render=False, seed=seed + i, **env_kwargs) for i in range(episodes)]
    for env in envs:
        env.set_difficulty(1.0)                      # the full task: no assist; perturbations are set below
        env.set_assist(assist)
        env.set_command(speed, heading)
        env.set_perturbation(perturbation, random_amplitude=False)
    obs = np.stack([env.reset() for env in envs])

    alive = np.ones(episodes, dtype=bool)
    fell = np.zeros(episodes, dtype=bool)
    ret = np.zeros(episodes)
    steps = np.zeros(episodes, dtype=np.int64)
    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        actions, _ = agent.select_action(obs[idx], training=stochastic)
        for j, i in enumerate(idx):
            obs[i], r, done, _ = envs[i].step(actions[j])
            ret[i] += r
            steps[i] += 1
            if done:
                fell[i], alive[i] = True, False

    stats = [env.episode_stats() for env in envs]
    out = {k: np.array([s[k] for s in stats]) for k in HumanoidRunEnv.STAT_KEYS}
    out.update(fell=fell, ret=ret, steps=steps, dt=envs[0].dt,
               success=np.array([env.episode_success(bool(f)) for env, f in zip(envs, fell)]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate running checkpoints under a fixed command.")
    parser.add_argument("--model", type=str, nargs="+", required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=[1.0, 2.0, 3.0], help="Commanded speeds (m/s)")
    parser.add_argument("--headings", type=float, nargs="+", default=[0.0], help="Heading offsets of the command (rad)")
    parser.add_argument("--perturbation", type=float, nargs="+", default=[0.0],
                        help="Intensity in [0, 1] of the reset randomization, domain randomization and pushes")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=1000, help="Episode cap in control steps (1000 = 15 s)")
    parser.add_argument("--assist", type=float, default=None,
                        help="Strength in [0, 1] of the stabilizing assist (default: none, the full task)")
    parser.add_argument("--stochastic", action="store_true", help="Sample the actions (default: use the mean)")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    torch.set_num_threads(1)
    for path in args.model:
        torch.manual_seed(args.seed)
        agent, ckpt, env_kwargs = load_agent(path)
        arch = ckpt.get("arch", {})
        if arch.get("task") != "run":
            raise SystemExit(f"{path} is not a running checkpoint (task: {arch.get('task', 'stand')})")
        print(f"\n=== {path}  (epoch {ckpt.get('epoch', '?')}, network {arch.get('hidden_layers')}, "
              f"{'stochastic' if args.stochastic else 'deterministic'})")
        print(f"{'v*':>5s} {'head':>5s} {'pert':>5s} | {'survive':>7s} {'success':>7s} {'time[s]':>7s} | {'speed':>5s} {'speed/v*':>8s} "
              f"{'lateral':>7s} {'head err':>8s} {'tilt':>5s} | {'dist [m]':>8s} {'return':>8s}")
        for speed in args.speeds:
            for heading in args.headings:
                for pert in args.perturbation:
                    r = run_command(agent, env_kwargs, speed, heading, pert, args.stochastic,
                                    args.episodes, args.max_steps, args.seed, args.assist)
                    surv = 1.0 - r["fell"].mean()
                    print(f"{speed:5.1f} {heading:5.2f} {pert:5.2f} | {surv:7.0%} {r['success'].mean():7.0%} {r['steps'].mean() * r['dt']:7.1f} | "
                          f"{r['speed'].mean():5.2f} {r['speed'].mean() / speed:8.2f} {r['lat'].mean():7.2f} "
                          f"{r['head_err'].mean():8.2f} {r['tilt'].mean():5.2f} | {r['dist'].mean():8.1f} {r['ret'].mean():8.0f}",
                          flush=True)


if __name__ == "__main__":
    main()
