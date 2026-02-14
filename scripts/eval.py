import argparse
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from envs.humanoid_env import HumanoidEnv

def evaluate_trajectory(file_path: str) -> None:
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Loading trajectory from {file_path}...")
    data = np.load(file_path)
    
    if 'qpos' not in data:
        print("Error: Invalid file format. 'qpos' key missing.")
        return

    qpos = data['qpos']
    z_heights = qpos[:, 2]
    
    try:
        env = HumanoidEnv(render=False)
        dt = env.model.opt.timestep
    except Exception as e:
        print(f"Warning: Could not load environment to get timestep ({e}). Defaulting to 0.002s")
        dt = 0.002

    fail_threshold = 0.2

    episodes_duration = []
    current_episode_steps = 0

    for z in z_heights:
        if z >= fail_threshold:
            current_episode_steps += 1
        else:
            if current_episode_steps > 0:
                episodes_duration.append(current_episode_steps * dt)
            current_episode_steps = 0
    
    if current_episode_steps > 0:
        episodes_duration.append(current_episode_steps * dt)

    if not episodes_duration:
        print("Result: The humanoid never stood up (z always < 0.2).")
        return

    print(f"--- Evaluation Report ---")
    print(f"Average standing time: {np.mean(episodes_duration):.2f} s")
    print(f"Stddev of standing time: {np.std(episodes_duration):.2f} s")
    print(f"Total episodes found:  {len(episodes_duration)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate standing time from a saved trajectory.")
    parser.add_argument("--input", type=str, help="Path to .npz trajectory file (e.g., data/humanoid_rollout.npz)", default="data/humanoid_rollout.npz")
    args = parser.parse_args()
    
    evaluate_trajectory(args.input)