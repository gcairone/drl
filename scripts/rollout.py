# replay.py
import argparse
import numpy as np
from envs.humanoid_env import HumanoidEnv
from agents.policy import NullPolicy


def run_rollout(env, policy, max_steps=1000):
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

    # converte in array numpy
    for k in trajectory:
        trajectory[k] = np.array(trajectory[k])

    return trajectory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/humanoid_rollout.npz")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    env = HumanoidEnv(render=False)
    policy = NullPolicy(env.action_dim)  # sostituire con policy reale

    traj = run_rollout(env, policy, max_steps=args.steps)
    np.savez(args.output, **traj)
    print(f"Rollout saved to {args.output}")
    print(f"Trajectory keys: {list(traj.keys())}")
    print(f"Trajectory lengths: {[len(traj[k]) for k in traj.keys()]}")
    print(f"dt_sim={env.model.opt.timestep:.6f}s")
    print(f"Total steps: {args.steps}, Total time: {args.steps * env.model.opt.timestep:.2f}s")

if __name__ == "__main__":
    main()
