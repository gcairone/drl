from typing import Dict
import time
import numpy as np
import argparse
from envs.humanoid_env import HumanoidEnv

def replay_trajectory(env: HumanoidEnv, trajectory: Dict[str, np.ndarray], fps: int = 60) -> None:
    """
    Replays a saved trajectory in the environment viewer.

    Args:
        env (HumanoidEnv): The environment instance.
        trajectory (Dict[str, np.ndarray]): Dictionary containing 'qpos' and 'qvel' arrays.
        fps (int): Frames per second for the playback.
    """
    dt_sim = env.model.opt.timestep
    skip = max(1, int(1 / fps / dt_sim))
    print(f"[INFO] dt_sim={dt_sim:.6f}s, fps={fps}, skip={skip} steps/frame")

    viewer = env.render_init_passive()
    n_steps = len(trajectory["qpos"])

    for i in range(0, n_steps, skip):
        qpos = trajectory["qpos"][i]
        qvel = trajectory["qvel"][i]

        # Update simulator state
        env._set_state(qpos, qvel)

        # Update GUI without advancing physics
        env.render_sync(state_only=True)

        msg = f"Time {i * dt_sim:.3f}s: step {i}"
        if "rewards" in trajectory:
            msg += f", reward={trajectory['rewards'][i]:.4f}"
        
        print(msg)

        time.sleep(1 / fps)  # Maintain stable frame rate

        if not viewer.is_running():
            print("[INFO] Viewer closed by user")
            break

    env.render_close()
    print("[INFO] Replay finished")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to trajectory npz", default="results/rollouts/humanoid_rollout.npz")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second for GUI")
    args = parser.parse_args()

    # Load trajectory
    data = np.load(args.file)
    trajectory = {"qpos": data["qpos"], "qvel": data["qvel"]}
    if "rewards" in data:
        trajectory["rewards"] = data["rewards"]

    # Create environment
    env = HumanoidEnv(render=True)

    # Replay
    replay_trajectory(env, trajectory, fps=args.fps)

if __name__ == "__main__":
    main()
