from typing import Dict, Optional
import time
import numpy as np
import argparse
from envs.humanoid_env import HumanoidEnv

def replay_trajectory(env: HumanoidEnv, trajectory: Dict[str, np.ndarray], fps: int = 60, dt: Optional[float] = None) -> None:
    """
    Replays a saved trajectory in the environment viewer.

    Args:
        env (HumanoidEnv): The environment instance.
        trajectory (Dict[str, np.ndarray]): Dictionary containing 'qpos' and 'qvel' arrays.
        fps (int): Frames per second for the playback.
        dt (float, optional): Time between saved states (control period). Defaults to the physics timestep.
    """
    dt_sim = dt if dt is not None else env.model.opt.timestep
    skip = max(1, int(1 / fps / dt_sim))
    print(f"[INFO] dt_sim={dt_sim:.6f}s, fps={fps}, skip={skip} steps/frame")

    viewer = env.render_init_passive()
    n_steps = len(trajectory["qpos"])

    frame_t = 1 / fps
    next_t = time.perf_counter()

    for i in range(0, n_steps, skip):
        qpos = trajectory["qpos"][i]
        qvel = trajectory["qvel"][i]

        env._set_state(qpos, qvel)

        env.render_sync(state_only=True)

        msg = f"Time {i * dt_sim:.3f}s: step {i}"
        if "rewards" in trajectory:
            msg += f", reward={trajectory['rewards'][i]:.4f}"
        
        if (i // skip) % fps == 0:
            print(msg)

        # keep real-time pace: sleep only for what the rendering did not already take
        next_t += frame_t
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()

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
    dt = float(data["dt"]) if "dt" in data else None
    replay_trajectory(env, trajectory, fps=args.fps, dt=dt)

if __name__ == "__main__":
    main()
