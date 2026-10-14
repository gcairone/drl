"""
Checks of the reset randomization, domain randomization, disturbances, curriculum and log_std floor.

    python3 -m scripts.test_randomization
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.ppo import PPOAgent
from envs.humanoid_env import HumanoidEnv, RANDOMIZATION_DEFAULTS
from models.actor_critic import GaussianActor
from utils.curriculum import DifficultyCurriculum


def test_difficulty_zero_is_the_plain_env() -> None:
    # reset must be exactly "initial state + uniform noise_scale noise", drawn from the env's own rng
    seed, noise = 11, 0.01
    env = HumanoidEnv(seed=seed, noise_scale=noise)
    ref = HumanoidEnv(seed=seed, noise_scale=noise)
    rng = np.random.default_rng(seed)
    for _ in range(3):
        obs = env.reset()
        qpos = ref.model.qpos0 + rng.uniform(-noise, noise, size=ref.model.nq)
        qvel = rng.uniform(-noise, noise, size=ref.model.nv)
        np.testing.assert_allclose(obs, np.concatenate([qpos, qvel]).astype(np.float32), atol=1e-6)
    assert env.intensity == 0.0 and not env.data.xfrc_applied.any()
    # no disturbance and no observation noise at difficulty 0, whatever the profile says
    env = HumanoidEnv(seed=1, push_interval=0.01)
    env.reset()
    for _ in range(200):
        env.step(np.zeros(env.action_dim))
        assert not env.data.xfrc_applied.any()


def test_dynamics_randomized_and_restored() -> None:
    env = HumanoidEnv(seed=2, difficulty=1.0, random_amplitude=False)
    nominal = {k: v.copy() for k, v in env._nominal.items()}
    env.reset()
    assert not np.array_equal(env.model.body_mass, nominal["body_mass"])
    assert not np.array_equal(env.model.actuator_gear, nominal["actuator_gear"])
    assert not np.array_equal(env.model.dof_damping, nominal["dof_damping"])
    assert not np.array_equal(env.model.geom_friction, nominal["geom_friction"])
    # inertia follows the mass, per body
    ratio = env.model.body_mass[1:] / nominal["body_mass"][1:]
    np.testing.assert_allclose(env.model.body_inertia[1:] / nominal["body_inertia"][1:], ratio[:, None] * np.ones((1, 3)), rtol=1e-9)
    # amplitudes respect the profile
    p = RANDOMIZATION_DEFAULTS
    assert np.all(np.abs(ratio - 1.0) <= p["dr_mass"] + 1e-9)
    assert np.all(np.abs(env.model.actuator_gear[:, 0] / nominal["actuator_gear"][:, 0] - 1.0) <= p["dr_gear"] + 1e-9)
    # back to difficulty 0: every parameter returns to its nominal value
    env.set_difficulty(0.0)
    env.reset()
    for key, value in nominal.items():
        assert np.array_equal(getattr(env.model, key), value), key


def test_reset_states_are_valid() -> None:
    env = HumanoidEnv(seed=3, difficulty=1.0, random_amplitude=False)
    tilts = []
    for _ in range(200):
        obs = env.reset()
        assert np.isfinite(obs).all()
        assert abs(np.linalg.norm(env.data.qpos[3:7]) - 1.0) < 1e-9
        n = env.data.ncon
        contact = env.data.contact
        on_floor = (contact.geom1[:n] == env._floor_geom) | (contact.geom2[:n] == env._floor_geom)
        if on_floor.any():
            assert -contact.dist[:n][on_floor].min() < 1e-3, "feet start inside the floor"
        assert env._self_penetration() <= HumanoidEnv.MAX_SELF_PENETRATION + 1e-9, "limbs start inside each other"
        assert env.data.qpos[2] < 1.282 + 0.1, "root lifted too much"
        tilts.append(2 * np.arcsin(np.linalg.norm(env.data.qpos[4:6])))  # roll/pitch part of the rotation
    assert max(tilts) > 0.02, "the tilt is not being randomized"
    assert max(tilts) <= RANDOMIZATION_DEFAULTS["rand_tilt"] * np.sqrt(2) + 0.02


def test_pushes() -> None:
    env = HumanoidEnv(seed=4, difficulty=1.0, random_amplitude=False, push_interval=0.2, push_force=100.0)
    env.reset()
    force_steps, starts, prev = 0, 0, 0.0
    lengths, current = [], 0
    for _ in range(600):
        env.step(np.zeros(env.action_dim))
        f = env.data.xfrc_applied[env._torso_id]
        magnitude = float(np.linalg.norm(f[:3]))
        assert magnitude <= 100.0 + 1e-9 and f[2] == 0.0 and not f[3:].any()
        if magnitude > 0:
            force_steps += 1
            current += 1
            starts += prev == 0
        elif current:
            lengths.append(current)
            current = 0
        prev = magnitude
        if env._is_done():
            env.reset()
            assert not env.data.xfrc_applied.any(), "force not cleared at reset"
            prev, current = 0.0, 0
    assert starts >= 5 and force_steps > 0
    expected = int(round(RANDOMIZATION_DEFAULTS["push_duration"] / env.dt))
    assert all(length == expected for length in lengths), (lengths, expected)


def test_random_amplitude() -> None:
    env = HumanoidEnv(seed=5, difficulty=0.6, random_amplitude=True)
    s = []
    for _ in range(300):
        env.reset()
        s.append(env.intensity)
    assert 0.0 <= min(s) and max(s) <= 0.6 and 0.25 < np.mean(s) < 0.35
    env.set_difficulty(0.6, random_amplitude=False)
    env.reset()
    assert env.intensity == 0.6


def test_observation_noise() -> None:
    env = HumanoidEnv(seed=6, difficulty=1.0, random_amplitude=False, rand_tilt=0.0, rand_joint_pos=0.0, rand_vel=0.0)
    obs = env.reset()
    clean = np.concatenate([env.data.qpos, env.data.qvel]).astype(np.float32)
    assert not np.array_equal(obs, clean) and np.abs(obs - clean).max() < 1.0


def test_curriculum() -> None:
    c = DifficultyCurriculum(start=0.1, step=0.1, up=0.8, down=0.4, min_episodes=10)
    for _ in range(9):
        c.record(True)
    assert not c.update() and c.level == 0.1, "must wait for min_episodes"
    c.record(True)
    assert c.update() and abs(c.level - 0.2) < 1e-9 and c.last_rate == 1.0
    for ok in [True] * 6 + [False] * 4:  # 60%: in the dead band
        c.record(ok)
    assert not c.update() and abs(c.level - 0.2) < 1e-9
    for ok in [True] * 3 + [False] * 7:  # 30%: too hard
        c.record(ok)
    assert c.update() and abs(c.level - 0.1) < 1e-9
    for _ in range(10):
        c.record(False)
    c.update()
    c.update()
    assert c.level == 0.0, "the level cannot go below 0"
    top = DifficultyCurriculum(start=0.95, step=0.1, min_episodes=1)
    top.record(True)
    top.update()
    assert top.level == 1.0
    saved = c.state_dict()
    c.record(True)
    other = DifficultyCurriculum()
    other.load_state_dict(saved)
    assert other.level == saved["level"] and len(other._results) == len(saved["results"])


def test_log_std_floor() -> None:
    floor = -1.2
    actor = GaussianActor(10, 4, log_std_min=floor)
    with torch.no_grad():
        actor.log_std[:] = torch.tensor([-3.0, -1.5, -1.0, 0.0])
    dist = actor._distribution(torch.zeros(5, 10))
    np.testing.assert_allclose(dist.stddev[0].log().detach().numpy(), [floor, floor, -1.0, 0.0], atol=1e-5)
    action, logp = actor.sample(torch.zeros(7, 10))
    np.testing.assert_allclose(logp.detach().numpy(), actor._distribution(torch.zeros(7, 10)).log_prob(action).sum(-1).detach().numpy(), atol=1e-4)

    # the stored parameter is held at the floor by the PPO update
    agent = PPOAgent(10, 4, log_std_min=floor, minibatch_size=16, update_epochs=2)
    with torch.no_grad():
        agent.actor.log_std[:] = -3.0
    n = 64
    obs = torch.randn(n, 10)
    with torch.no_grad():
        act, logp_old = agent.actor.sample(obs)

    class FakeBuffer:
        def get(self):
            return dict(obs=obs, act=act, ret=torch.randn(n), adv=torch.randn(n), logp=logp_old)

    agent.update(FakeBuffer())
    assert agent.actor.log_std.min().item() >= floor - 1e-6
    # without a floor the parameter is left alone
    free = GaussianActor(10, 4)
    with torch.no_grad():
        free.log_std[:] = -3.0
    assert free._distribution(torch.zeros(1, 10)).stddev.log().max().item() < -2.9


def test_adaptive_lr() -> None:
    def make(adaptive: bool, lr: float) -> PPOAgent:
        torch.manual_seed(0)
        return PPOAgent(10, 4, lr=lr, minibatch_size=16, update_epochs=3, target_kl=0.01, adaptive_lr=adaptive,
                        lr_max=max(lr, 5e-4))

    n = 64
    obs = torch.randn(n, 10)
    adv_big, adv_zero = torch.randn(n), torch.zeros(n)

    def run(agent: PPOAgent, adv: torch.Tensor) -> dict:
        with torch.no_grad():
            act, logp = agent.actor.sample(obs)

        class FakeBuffer:
            def get(self):
                return dict(obs=obs, act=act, ret=torch.randn(n), adv=adv, logp=logp)

        return agent.update(FakeBuffer())

    # a zero advantage keeps the policy still: the KL stays below the target and the rate climbs to its ceiling
    calm = make(True, 1e-4)
    out = run(calm, adv_zero)
    assert out["lr"] > 1e-4 and out["update_epochs"] == 3
    for _ in range(10):
        out = run(calm, adv_zero)
    assert out["lr"] == calm.lr_max
    # a too-large rate moves the policy far beyond the target: it is cut down, and all the passes are still done
    hard = make(True, 3e-2)
    for _ in range(6):
        out = run(hard, adv_big)
    assert out["lr"] < 3e-3 and out["update_epochs"] == 3, out
    # without it the rate is untouched and the KL early stop is still there
    fixed = make(False, 3e-2)
    out = run(fixed, adv_big)
    assert out["lr"] == 3e-2 and out["update_epochs"] < 3, out


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
