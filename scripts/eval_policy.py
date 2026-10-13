import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.loading import load_agent  # noqa: F401  (re-exported for the other scripts)
from agents.ppo import PPOAgent
from envs.humanoid_env import HumanoidEnv


def run_episodes(agent: PPOAgent, env_kwargs: Dict, noise_scale: float, stochastic: bool,
                 episodes: int, max_steps: int, seed: int, difficulty: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Runs `episodes` independent episodes in lockstep (one fresh env each, different seeds),
    with one batched forward pass per step over the envs still alive.
    `difficulty` is the randomization level, applied at its exact value (see HumanoidEnv.set_difficulty).

    Returns:
        Dict with 'steps' (episode length per env), 'ret' (undiscounted return per env)
        and 'fell' (True if the episode ended by termination, False if it reached max_steps).
    """
    kwargs = dict(env_kwargs, noise_scale=noise_scale)
    envs = [HumanoidEnv(render=False, seed=seed + i, difficulty=difficulty, random_amplitude=False, **kwargs)
            for i in range(episodes)]
    obs = np.stack([env.reset() for env in envs])

    steps = np.zeros(episodes, dtype=np.int64)
    ret = np.zeros(episodes)
    fell = np.zeros(episodes, dtype=bool)
    alive = np.ones(episodes, dtype=bool)

    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        actions, _ = agent.select_action(obs[idx], training=stochastic)
        for j, i in enumerate(idx):
            o, r, done, _ = envs[i].step(actions[j])
            obs[i] = o
            ret[i] += r
            steps[i] += 1
            if done:
                fell[i] = True
                alive[i] = False
    return {"steps": steps, "ret": ret, "fell": fell, "dt": envs[0].dt}


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    """95% Wilson interval for a proportion."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(res: Dict[str, np.ndarray], max_steps: int) -> Dict[str, float]:
    t = res["steps"] * res["dt"]
    n = len(t)
    survived = int((~res["fell"]).sum())
    lo, hi = wilson(survived, n)
    return {
        "mean": t.mean(), "std": t.std(ddof=1) if n > 1 else 0.0,
        "sem": t.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
        "median": np.median(t), "p10": np.percentile(t, 10), "min": t.min(),
        "surv": survived / n, "surv_lo": lo, "surv_hi": hi, "ret": res["ret"].mean(),
        "cap": max_steps * res["dt"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints over many independent episodes.")
    parser.add_argument("--model", type=str, nargs="+", required=True, help="One or more PPO checkpoints")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per configuration")
    parser.add_argument("--max_steps", type=int, default=1000, help="Episode cap in control steps")
    parser.add_argument("--noise", type=float, nargs="+", default=[0.01, 0.05, 0.1],
                        help="Reset noise scales to test (training value is the first column of interest)")
    parser.add_argument("--difficulty", type=float, nargs="+", default=[0.0],
                        help="Randomization levels in [0, 1] to test (resets, domain randomization, pushes); 0 is the plain environment")
    parser.add_argument("--modes", type=str, nargs="+", default=["det", "stoch"], choices=["det", "stoch"],
                        help="det = mean action, stoch = sampled actions as in training")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed (envs use seed+i); disjoint from training seeds")
    args = parser.parse_args()

    torch.set_num_threads(1)
    for path in args.model:
        torch.manual_seed(args.seed)
        agent, ckpt, env_kwargs = load_agent(path)

        print(f"\n=== {path}  (epoch {ckpt.get('epoch', '?')}, env {env_kwargs})")
        print(f"{'mode':5s} {'noise':>6s} {'diff':>5s} | {'mean±sem [s]':>14s} {'std':>5s} {'med':>6s} {'p10':>6s} {'min':>6s} | "
              f"{'survive cap':>11s} {'95% CI':>13s} | {'return':>7s}")
        for noise in args.noise:
          for difficulty in args.difficulty:
            for mode in args.modes:
                res = run_episodes(agent, env_kwargs, noise, mode == "stoch", args.episodes, args.max_steps, args.seed, difficulty)
                s = summarize(res, args.max_steps)
                print(f"{mode:5s} {noise:6.2f} {difficulty:5.2f} | {s['mean']:6.2f}±{s['sem']:5.2f}   {s['std']:5.2f} {s['median']:6.2f} "
                      f"{s['p10']:6.2f} {s['min']:6.2f} | {s['surv']:11.0%} [{s['surv_lo']:.2f},{s['surv_hi']:.2f}] | {s['ret']:7.1f}",
                      flush=True)
        print(f"(cap = {s['cap']:.1f} s, {args.episodes} episodes per row)")


if __name__ == "__main__":
    main()
