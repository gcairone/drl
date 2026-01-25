# scripts/play.py
import time
import numpy as np
import argparse
from envs.humanoid_env import HumanoidEnv

def replay_trajectory(env, trajectory, fps=60):
    dt_sim = env.model.opt.timestep
    skip = max(1, int(1 / fps / dt_sim))
    print(f"[INFO] dt_sim={dt_sim:.6f}s, fps={fps}, skip={skip} steps/frame")

    viewer = env.render_init_passive()
    n_steps = len(trajectory["qpos"])

    for i in range(0, n_steps, skip):
        qpos = trajectory["qpos"][i]
        qvel = trajectory["qvel"][i]

        # Aggiorna stato del simulatore
        env._set_state(qpos, qvel)

        # Aggiorna GUI senza far avanzare la fisica
        env.render_sync(state_only=True)

        print(f"Time {i * dt_sim:.3f}s: step {i}")

        time.sleep(1 / fps)  # mantiene frame rate stabile

        if not viewer.is_running():
            print("[INFO] Viewer closed by user")
            break

    env.render_close()
    print("[INFO] Replay finished")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to trajectory npz", default="data/humanoid_rollout.npz")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second for GUI")
    args = parser.parse_args()

    # Carica la traiettoria
    data = np.load(args.file)
    trajectory = {"qpos": data["qpos"], "qvel": data["qvel"]}

    # Crea env
    env = HumanoidEnv(render=True)

    # Replay
    replay_trajectory(env, trajectory, fps=args.fps)

if __name__ == "__main__":
    main()
