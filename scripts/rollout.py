from typing import Dict, Any
import argparse
import numpy as np
import torch
from envs.humanoid_env import HumanoidEnv
from agents.policy import BasePolicy, RandomPolicy
from agents.ppo import PPOAgent


def run_rollout(env: HumanoidEnv, policy: BasePolicy, max_steps: int = 1000) -> Dict[str, np.ndarray]:
    """
    Executes a simulation rollout using the given policy.

    Args:
        env (HumanoidEnv): The environment instance.
        policy (BasePolicy): The policy to use for action selection.
        max_steps (int): Maximum number of steps to simulate.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the trajectory data (qpos, qvel, actions, rewards).
    """
    obs = env.reset()
    trajectory = {
        "qpos": [],
        "qvel": [],
        "actions": [],
        "rewards": [],
    }

    for step in range(max_steps):
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)

        trajectory["qpos"].append(env.data.qpos.copy())
        trajectory["qvel"].append(env.data.qvel.copy())
        trajectory["actions"].append(action.copy())
        trajectory["rewards"].append(reward)

        if done:
            obs = env.reset()

    # Convert lists to numpy arrays
    for k in trajectory:
        trajectory[k] = np.array(trajectory[k])

    return trajectory

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/rollouts/humanoid_rollout.npz")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--model", type=str, default=None, help="Path to PPO checkpoint")
    args = parser.parse_args()

    env = HumanoidEnv(render=False)
    
    if args.model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading PPO model from {args.model} on {device}")
        agent = PPOAgent(state_dim=env.obs_dim, action_dim=env.action_dim, device=device)
        agent.load(args.model)
        policy = agent.get_policy()
    else:
        print("Using RandomPolicy")
        policy = RandomPolicy(env.action_dim)

    traj = run_rollout(env, policy, max_steps=args.steps)
    np.savez(args.output, **traj)
    print(f"Rollout saved to {args.output}")
    print(f"Trajectory keys: {list(traj.keys())}")
    print(f"Trajectory lengths: {[len(traj[k]) for k in traj.keys()]}")
    print(f"dt_sim={env.model.opt.timestep:.6f}s")
    print(f"Total steps: {args.steps}, Total time: {args.steps * env.model.opt.timestep:.2f}s")

if __name__ == "__main__":
    main()
