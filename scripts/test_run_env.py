"""
Checks of the running task.

    python3 -m scripts.test_run_env
"""
import os
import sys

import mujoco
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from envs.humanoid_env import HumanoidEnv
from envs.run_env import (HumanoidRunEnv, RUN_DEFAULTS, assist_level, commanded_speed_range, heading_range,
                          perturbation_level, wrap_angle)


def settle(env: HumanoidRunEnv, steps: int = 40, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        env.step(rng.uniform(-1, 1, env.action_dim) * 0.2)


def test_observation_and_determinism() -> None:
    a, b = HumanoidRunEnv(seed=5), HumanoidRunEnv(seed=5)
    oa, ob = a.reset(), b.reset()
    assert oa.shape == (a.obs_dim,) == (76,) and np.isfinite(oa).all()
    np.testing.assert_array_equal(oa, ob)
    for _ in range(30):
        act = np.random.default_rng(1).uniform(-1, 1, a.action_dim)
        oa, ra, da, _ = a.step(act)
        ob, rb, db, _ = b.step(act)
        assert np.array_equal(oa, ob) and ra == rb and da == db
    # upright at the start: gravity points down in the torso frame
    c = HumanoidRunEnv(seed=1)
    obs = c.reset()
    np.testing.assert_allclose(obs[1:4], [0, 0, -1], atol=0.05)
    assert abs(obs[-3] - 1.0) < 1e-6 and abs(obs[-2]) < 1e-6 and abs(obs[-1] - c._v_target) < 1e-6  # command: straight ahead at v*


def test_observation_is_invariant_to_the_heading_of_the_world() -> None:
    env = HumanoidRunEnv(seed=2)
    env.reset()
    env.set_command(2.0, 0.4)
    env.reset()
    settle(env)
    before = env._get_obs()
    for theta in (0.7, -2.0, 3.0):
        qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        qpos2, qvel2 = qpos.copy(), qvel.copy()
        qpos2[:2] = rot @ qpos[:2]
        qvel2[:2] = rot @ qvel[:2]
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)]), qpos[3:7])
        qpos2[3:7] = quat
        cmd = env._cmd_yaw
        env._set_state(qpos2, qvel2)
        env._cmd_yaw = wrap_angle(cmd + theta)
        np.testing.assert_allclose(env._get_obs(), before, atol=1e-5)
        env._set_state(qpos, qvel)
        env._cmd_yaw = cmd


def test_frames() -> None:
    env = HumanoidRunEnv(seed=3)
    env.reset()
    env.data.qvel[:] = 0.0
    # world velocity along +x with the torso facing +x: forward in the torso frame
    env.data.qvel[0] = 1.5
    env._refresh_kinematics()
    obs = env._get_obs()
    np.testing.assert_allclose(obs[4:7], [1.5, 0, 0], atol=0.05)
    # facing +y (yaw 90 degrees): the same world velocity is now to the right of the torso (-y of the body)
    qpos = env.data.qpos.copy()
    qpos[3:7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    env._set_state(qpos, env.data.qvel.copy())
    obs = env._get_obs()
    np.testing.assert_allclose(obs[4:7], [0, -1.5, 0], atol=0.05)


def test_reward() -> None:
    env = HumanoidRunEnv(seed=4)
    env.reset()
    env.set_command(2.0, 0.0)
    env.reset()
    zero = np.zeros(env.action_dim, dtype=np.float32)

    def step_reward(v_par: float, v_perp: float = 0.0) -> float:
        env._prev_action[:] = 0.0
        env.data.qvel[:] = 0.0
        env.data.qvel[0], env.data.qvel[1] = v_par, v_perp
        return env._observe_step(zero)[1]

    rewards = [step_reward(v) for v in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert all(x < y for x, y in zip(rewards, rewards[1:])), rewards  # closer to the target is better
    assert step_reward(2.0) > step_reward(2.8) and step_reward(2.0) > step_reward(3.5)  # overshooting is worse
    assert step_reward(2.0, v_perp=1.0) < step_reward(2.0)  # lateral drift is penalized
    assert step_reward(-1.0) < step_reward(0.0)  # running backwards is worse than standing
    # a perfect step: every shaping term is 1 and no cost
    p = RUN_DEFAULTS
    np.testing.assert_allclose(step_reward(2.0), p["w_alive"] + p["w_vel"] + p["w_prog"] + p["w_head"] + p["w_up"], atol=0.02)
    # efforts cost
    env.data.qvel[:] = 0.0
    env.data.qvel[0] = 2.0
    env._prev_action[:] = 0.0
    cost = env._observe_step(np.ones(env.action_dim, dtype=np.float32))[1]
    env.data.qvel[:] = 0.0
    env.data.qvel[0] = 2.0
    env._prev_action[:] = 0.0
    assert cost < env._observe_step(zero)[1]


def test_termination() -> None:
    env = HumanoidRunEnv(seed=6)
    env.reset()
    zero = np.zeros(env.action_dim, dtype=np.float32)
    assert not env._observe_step(zero)[2]
    qpos = env.data.qpos.copy()
    qpos[2] = RUN_DEFAULTS["fall_height"] - 0.05
    env._set_state(qpos, env.data.qvel.copy())
    assert env._observe_step(zero)[2], "too low"
    env.reset()
    qpos = env.data.qpos.copy()
    tilt = RUN_DEFAULTS["fall_tilt"] + 0.2
    qpos[3:7] = [np.cos(tilt / 2), 0, np.sin(tilt / 2), 0]  # pitched forward
    env._set_state(qpos, env.data.qvel.copy())
    assert env._observe_step(zero)[2], "too tilted"


def test_curriculum_schedule() -> None:
    p = RUN_DEFAULTS
    assert commanded_speed_range(0.0, p) == (p["run_speed_min"], p["run_speed_start"])
    assert commanded_speed_range(1.0, p) == (p["run_speed_min"], p["run_speed"])
    assert commanded_speed_range(0.3, p)[1] < commanded_speed_range(0.5, p)[1]
    assert heading_range(0.3, p) == 0.0 and abs(heading_range(1.0, p) - p["run_heading"]) < 1e-9
    assert perturbation_level(0.4) == 0.0 and perturbation_level(1.0) == 1.0 and 0 < perturbation_level(0.7) < 1
    # sampled commands respect the ranges, and the env follows set_difficulty
    env = HumanoidRunEnv(seed=7)
    for level in (0.0, 0.5, 1.0):
        env.set_difficulty(level)
        speeds, headings = [], []
        for _ in range(150):
            env.reset()
            speeds.append(env._v_target)
            headings.append(wrap_angle(env._cmd_yaw))
        lo, hi = commanded_speed_range(level, p)
        assert lo - 1e-9 <= min(speeds) and max(speeds) <= hi + 1e-9
        assert max(speeds) > lo + 0.6 * (hi - lo), "the commanded speed does not span its range"
        assert max(np.abs(headings)) <= heading_range(level, p) + 0.05
        h = heading_range(level, p)
        assert (max(np.abs(headings)) > 0.5 * h) if h > 0 else (max(np.abs(headings)) < 0.05), (level, h)
    # perturbation follows the level, and set_perturbation overrides it
    env.set_difficulty(0.2)
    env.reset()
    assert env.intensity == 0.0
    env.set_difficulty(1.0, random_amplitude=False)
    env.reset()
    assert env.intensity == 1.0
    env.set_perturbation(0.3)
    env.reset()
    assert env.intensity == 0.3
    # a fixed command wins over the sampling
    env.set_command(2.5, -0.5)
    env.reset()
    assert env._v_target == 2.5 and abs(wrap_angle(env._cmd_yaw) + 0.5) < 0.05


def test_episode_outcome() -> None:
    env = HumanoidRunEnv(seed=8)
    env.set_command(2.0, 0.0)
    env.reset()
    zero = np.zeros(env.action_dim, dtype=np.float32)
    assert not env.episode_success(fell=False), "an episode with no step cannot succeed"
    for v in (1.9, 2.1, 2.0, 1.8):
        env.data.qvel[:] = 0.0
        env.data.qvel[0] = v
        env._observe_step(zero)
    stats = env.episode_stats()
    assert set(HumanoidRunEnv.STAT_KEYS) <= set(stats)
    assert abs(stats["speed"] - 1.95) < 0.02 and stats["v_target"] == 2.0
    assert abs(stats["dist"] - 7.8 * env.dt) < 0.02
    assert env.episode_success(fell=False) and not env.episode_success(fell=True)
    env.reset()
    for _ in range(4):
        env.data.qvel[:] = 0.0
        env.data.qvel[0] = 1.0  # half the commanded speed: below success_speed
        env._observe_step(zero)
    assert not env.episode_success(fell=False)
    # the standing task keeps the old outcome rule
    plain = HumanoidEnv(seed=0)
    assert plain.episode_success(fell=False) and not plain.episode_success(fell=True) and plain.episode_stats() == {}


def test_disturbances_and_domain_randomization_are_inherited() -> None:
    env = HumanoidRunEnv(seed=9, push_interval=0.2)
    env.set_command(1.0, 0.0)
    env.set_perturbation(1.0)
    env.set_assist(0.0)
    nominal = env._nominal["body_mass"].copy()
    env.reset()
    assert not np.array_equal(env.model.body_mass, nominal)
    pushes = 0
    prev = 0.0
    for _ in range(300):
        env.step(np.zeros(env.action_dim))
        f = float(np.linalg.norm(env.data.xfrc_applied[env._torso_id, :2]))  # horizontal: the assist is vertical
        pushes += f > 0 and prev == 0
        prev = f
        if env._is_done():
            env.reset()
            prev = 0.0
    assert pushes >= 3
    env.set_perturbation(0.0)
    env.reset()
    assert np.array_equal(env.model.body_mass, nominal)
    obs = env._get_obs()
    assert np.array_equal(obs, env._get_obs()), "no observation noise without perturbation"


def test_assist() -> None:
    p = RUN_DEFAULTS
    assert assist_level(0.0, p) == 1.0 and assist_level(p["assist_end"], p) == 0.0 and assist_level(1.0, p) == 0.0
    assert 0 < assist_level(p["assist_end"] / 2, p) < 1
    env = HumanoidRunEnv(seed=11)
    weight = float(env.model.body_mass.sum()) * 9.81
    for level, expected in ((0.0, 1.0), (p["assist_end"] / 2, 0.5), (p["assist_end"], 0.0), (1.0, 0.0)):
        env.set_difficulty(level)
        env.reset()
        env.step(np.zeros(env.action_dim))
        f = env.data.xfrc_applied[env._torso_id]
        assert abs(env._assist - expected) < 1e-9
        assert abs(f[2] - expected * p["assist_support"] * weight) < 1e-6, (level, f[2])
        if expected == 0.0:
            assert not f[2:6].any(), "no assist force or torque once the assist is gone"
    # the override wins over the schedule and a fixed strength is kept
    env.set_difficulty(1.0)
    env.set_assist(0.4)
    env.reset()
    assert env._assist == 0.4
    env.set_assist(None)
    env.reset()
    assert env._assist == 0.0
    # the torque pulls a tilted torso back to the vertical and does not push a vertical one
    env.set_assist(1.0)
    env.reset()
    qpos = env.data.qpos.copy()
    qpos[3:7] = [np.cos(0.1), 0, np.sin(0.1), 0]          # pitched by 0.2 rad about y: up axis leans towards +x
    env._set_state(qpos, np.zeros_like(env.data.qvel))
    env._apply_assist()
    torque = env.data.xfrc_applied[env._torso_id, 3:6].copy()
    assert torque[1] < 0 and abs(torque[0]) < 1e-9 and torque[2] == 0.0, torque   # restoring: rotates back about -y
    env.reset()
    env._apply_assist()
    np.testing.assert_allclose(env.data.xfrc_applied[env._torso_id, 3:6], 0.0, atol=15.0)  # nearly upright: ~no torque
    # the episode statistics report the strength used
    env.set_assist(0.6)
    env.reset()
    env.step(np.zeros(env.action_dim))
    assert "assist" in HumanoidRunEnv.STAT_KEYS and abs(env.episode_stats()["assist"] - 0.6) < 1e-9


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
