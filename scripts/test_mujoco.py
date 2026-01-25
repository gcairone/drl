# scripts/test_mujoco.py
import time
import numpy as np
from envs.humanoid_env import HumanoidEnv
from agents.policy import NullPolicy
import mujoco



def main():
    # Crea environment con rendering
    env = HumanoidEnv(render=True)

    policy = NullPolicy(env.action_dim)
    step = 0

    env.reset()

    def control_callback(model, data):
        nonlocal step
        action = policy(env._get_obs())
        data.ctrl[:] = action
        step += 1

    mujoco.set_mjcb_control(control_callback)

    try:
        env.viewer = mujoco.viewer.launch(env.model, env.data)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
