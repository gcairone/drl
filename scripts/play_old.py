# play.py
import argparse
import numpy as np
import time
from envs.humanoid_env_old import HumanoidEnv

def replay_with_fps(env, trajectory, fps=60):
    """
    Replay della traiettoria usando GUI a FPS scelti.
    skip viene calcolato in base al dt della simulazione.
    """
    dt_sim = env.model.opt.timestep
    dt_frame = 1.0 / fps
    skip = max(1, round(dt_frame / dt_sim))
    print(f"dt_sim={dt_sim:.6f}s, fps={fps}, skip={skip} steps/frame")

    env.render_mode = True
    for i in range(0, len(trajectory["qpos"]), skip):
        qpos = trajectory["qpos"][i]
        qvel = trajectory["qvel"][i]
        env._set_state(qpos, qvel)
        print(f"Time {i * dt_sim:.2f}s: step {i}")
        env.render_step()

        
        time.sleep(skip * dt_sim)  # tempo reale corrispondente

    env.close()
    print("Replay finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="humanoid_rollout.npz")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    traj_data = np.load(args.input)
    trajectory = {k: traj_data[k] for k in traj_data.keys()}

    env = HumanoidEnv(render=True)
    replay_with_fps(env, trajectory, fps=args.fps)

if __name__ == "__main__":
    main()
