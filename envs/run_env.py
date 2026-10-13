"""
Running task: the humanoid has to run along a commanded horizontal direction at a commanded speed.

Command. For every episode a speed v* (m/s) and a direction are drawn. The direction is a heading
offset phi from the initial facing direction of the torso, so phi = 0 means "straight ahead".

Observation (no absolute position or heading, so it does not grow with the distance covered and
the policy cannot depend on where it is): torso height, gravity direction, linear velocity (whole-body
centre of mass) and angular velocity in the torso frame, joint angles, joint velocities, the previous
action, and the command (cos/sin of the angle between the torso heading and the commanded direction,
and v*).

Reward (per control step), with v_par / v_perp the centre-of-mass velocity along / across the
commanded direction:

    w_alive
  + w_vel  * exp(-((v_par - v*) / sigma)^2)      precise speed tracking, sigma = max(0.3, 0.35 v*)
  + w_prog * clip(v_par / v*, 0, 1) * min(1, v*/0.5)   dense progress when still far from the target (none if v* = 0)
  + w_head * (1 + cos(heading error)) / 2         face the commanded direction
  + w_up   * exp(-(tilt / 0.5)^2)                 keep the trunk reasonably upright
  - w_lat  * min(v_perp^2, 4)                     do not drift sideways
  - w_ctrl * |u|^2 - w_rate * |a - a_prev|^2      effort (u = applied motor control) and smooth actions

The episode ends when the torso falls below `fall_height` or tilts beyond `fall_tilt`.

Curriculum. A single level in [0, 1] (see set_difficulty) drives four things, one after the other:
  1. an assist, as training wheels: a stabilizing torque that keeps the trunk upright and an upward force
     that carries most of the body weight (a harness).
     It is full at level 0 and gone at `assist_end`; without it the policy settles for falling forward
     at the commanded speed instead of learning to balance;
  2. the upper bound of the commanded speed, from a slow walk up to `run_speed` (levels 0.1 to 0.7);
  3. the heading offset of the command (from level 0.4);
  4. the intensity of the reset randomization, domain randomization and pushes inherited from
     HumanoidEnv (from level 0.5). An episode counts as a success for the curriculum when
the humanoid does not fall and its average speed along the command reaches `success_speed` times v*.
"""
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

from envs.humanoid_env import HumanoidEnv

RUN_DEFAULTS: Dict[str, float] = dict(
    # --- command
    run_speed=3.5,         # m/s: upper bound of the commanded speed at the end of the curriculum
    run_speed_min=0.0,     # m/s: lower bound of the commanded speed (0 = stand still)
    run_speed_start=0.5,   # m/s: upper bound of the commanded speed at level 0
    run_heading=0.8,       # rad: largest heading offset of the command at the end of the curriculum
    # --- reward weights
    w_alive=0.5,
    w_vel=2.0,
    w_prog=0.5,
    w_head=0.5,
    w_up=0.3,
    w_lat=0.5,
    w_ctrl=0.005,
    w_rate=0.01,
    # --- action space: 1 = the action is a target joint angle tracked by a PD controller, 0 = raw motor torques
    pd_control=1.0,
    pd_kp=4.0,             # PD stiffness, in motor control units (full torque) per radian of error
    pd_kd=0.2,             # PD damping, in motor control units per rad/s
    pd_scale=0.5,          # target = action * pd_scale * (largest |range| of the joint), within [0.3, 1.0] rad
    # --- assist (training wheels of the curriculum)
    assist_kp=600.0,       # N m/rad: stiffness of the stabilizing torque on the torso at full assist
    assist_kd=60.0,        # N m s/rad: damping of the stabilizing torque
    assist_support=0.9,    # share of the body weight carried by an upward force on the torso at full assist
    assist_end=0.25,       # curriculum level at which the assist is gone (0: no assist)
    # --- termination and success
    fall_height=0.8,       # m: torso height below which the episode ends
    fall_tilt=1.0,         # rad: torso tilt beyond which the episode ends
    success_speed=0.7,     # share of the commanded speed to reach (on average) for a successful episode
)

# levels of the curriculum at which each aspect starts / finishes
SPEED_RAMP = (0.1, 0.7)
HEADING_START = 0.4
PERTURBATION_START = 0.5


def commanded_speed_range(level: float, run: Dict[str, float]) -> Tuple[float, float]:
    """Interval the commanded speed is drawn from at a given curriculum level."""
    ramp = float(np.clip((level - SPEED_RAMP[0]) / (SPEED_RAMP[1] - SPEED_RAMP[0]), 0.0, 1.0))
    hi = run["run_speed_start"] + (run["run_speed"] - run["run_speed_start"]) * ramp
    return run["run_speed_min"], max(hi, run["run_speed_min"])


def assist_level(level: float, run: Dict[str, float]) -> float:
    """Strength of the stabilizing torque in [0, 1] at a given curriculum level."""
    if run["assist_end"] <= 0:
        return 0.0
    return float(np.clip(1.0 - level / run["assist_end"], 0.0, 1.0))


def heading_range(level: float, run: Dict[str, float]) -> float:
    """Largest heading offset |phi| of the command at a given curriculum level."""
    return run["run_heading"] * float(np.clip((level - HEADING_START) / (1.0 - HEADING_START), 0.0, 1.0))


def perturbation_level(level: float) -> float:
    """Level of the randomization inherited from HumanoidEnv at a given curriculum level."""
    return float(np.clip((level - PERTURBATION_START) / (1.0 - PERTURBATION_START), 0.0, 1.0))


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class HumanoidRunEnv(HumanoidEnv):
    """Run along a commanded direction at a commanded speed. See the module docstring."""

    STAT_KEYS = ("speed", "v_target", "assist", "dist", "lat", "head_err", "tilt",
                 "r_vel", "r_prog", "r_head", "r_up", "ctrl", "rate")

    def __init__(self, seed: Optional[int] = None, difficulty: float = 0.0, random_amplitude: bool = True,
                 **kwargs: Any) -> None:
        """
        Args:
            difficulty (float): Curriculum level in [0, 1], see set_difficulty().
            **kwargs: Overrides of RUN_DEFAULTS, plus any argument of HumanoidEnv (including the
                      randomization profile).
        """
        run = {k: kwargs.pop(k) for k in list(kwargs) if k in RUN_DEFAULTS}
        self.run: Dict[str, float] = {**RUN_DEFAULTS, **run}
        self.level = 0.0
        self._fixed_command: Optional[Tuple[float, float]] = None
        self._assist_override: Optional[float] = None
        self._assist = 0.0
        super().__init__(seed=seed, difficulty=difficulty, random_amplitude=random_amplitude, **kwargs)

        nu = self.action_dim
        self.obs_dim = 1 + 3 + 3 + 3 + 3 * nu + 3
        # observation noise: x1 on positions and directions, x10 on velocities, none on the action and the command
        self._run_sigma_unit = np.concatenate([
            [1.0], np.ones(3), 10.0 * np.ones(3), 10.0 * np.ones(3),
            np.ones(nu), 10.0 * np.ones(nu), np.zeros(nu), np.zeros(3),
        ]).astype(np.float32)
        self.rng_cmd = np.random.default_rng(None if seed is None else seed + 104729)
        self._prev_action = np.zeros(nu, dtype=np.float32)
        self._v_target = 0.0
        self._cmd_yaw = 0.0
        self._reset_accumulators()

        # PD control: actuator i drives joint trnid[i]; the amplitude of the target follows the joint range
        joints = self.model.actuator_trnid[:, 0]
        self._act_qpos = self.model.jnt_qposadr[joints].copy()
        self._act_dof = self.model.jnt_dofadr[joints].copy()
        reach = np.abs(self.model.jnt_range[joints]).max(axis=1)
        self._pd_amp = np.clip(self.run["pd_scale"] * reach, 0.3, 1.0)
        self._pd_target = np.zeros(nu)

    # -------------------
    # Control
    # -------------------
    def _set_action(self, action: np.ndarray) -> None:
        if self.run["pd_control"] > 0.5:
            self._pd_target = action * self._pd_amp
            self._apply_control()
        else:
            self.data.ctrl[:] = action

    def _apply_control(self) -> None:
        if self.run["pd_control"] > 0.5:
            q = self.data.qpos[self._act_qpos]
            qd = self.data.qvel[self._act_dof]
            self.data.ctrl[:] = np.clip(self.run["pd_kp"] * (self._pd_target - q) - self.run["pd_kd"] * qd, -1.0, 1.0)
        if self._assist > 0.0:
            self._apply_assist()

    def _apply_assist(self) -> None:
        """Torque on the torso pulling its up axis back to the vertical, damped on the horizontal angular velocity."""
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        R = R.reshape(3, 3)
        up = R[:, 2]                                   # torso up axis in the world frame
        w = R @ self.data.qvel[3:6]                    # angular velocity in the world frame
        torque = np.array([up[1], -up[0], 0.0]) * self.run["assist_kp"] - np.array([w[0], w[1], 0.0]) * self.run["assist_kd"]
        self.data.xfrc_applied[self._torso_id, 3:6] = self._assist * torque
        # weight support: the legs carry only the rest of the weight (the pushes use the horizontal force components)
        weight = float(self.model.body_mass.sum()) * -float(self.model.opt.gravity[2])
        self.data.xfrc_applied[self._torso_id, 2] = self._assist * self.run["assist_support"] * weight

    # -------------------
    # Curriculum and evaluation controls
    # -------------------
    def set_difficulty(self, difficulty: float, random_amplitude: bool = True) -> None:
        """
        Sets the curriculum level in [0, 1]; it takes effect from the next reset(). The level fixes the
        range the commanded speed and heading offset are drawn from, and (through HumanoidEnv) the
        intensity of the randomization and of the pushes.
        """
        self.level = float(np.clip(difficulty, 0.0, 1.0))
        super().set_difficulty(perturbation_level(self.level), random_amplitude)

    def set_assist(self, strength: Optional[float]) -> None:
        """Fixes the strength of the assist in [0, 1] from the next reset on; None restores the curriculum schedule."""
        self._assist_override = None if strength is None else float(np.clip(strength, 0.0, 1.0))

    def set_command(self, speed: Optional[float], heading: float = 0.0) -> None:
        """Fixes the command of the next episodes (speed in m/s, heading offset in rad); None restores the sampling."""
        self._fixed_command = None if speed is None else (float(speed), float(heading))

    def set_perturbation(self, level: float, random_amplitude: bool = False) -> None:
        """Sets the intensity of randomization and pushes directly, independently of the curriculum level."""
        super().set_difficulty(level, random_amplitude)

    # -------------------
    # Episode
    # -------------------
    def reset(self) -> np.ndarray:
        super().reset()
        self._prev_action[:] = 0.0
        self._init_episode()
        return self._get_obs()

    def _init_episode(self) -> None:
        R = self._rotation()
        yaw0 = float(np.arctan2(R[1, 0], R[0, 0]))
        if self._fixed_command is not None:
            speed, heading = self._fixed_command
        else:
            lo, hi = commanded_speed_range(self.level, self.run)
            speed = self.rng_cmd.uniform(lo, hi)
            heading = self.rng_cmd.uniform(-1.0, 1.0) * heading_range(self.level, self.run)
        self._v_target = float(speed)
        self._cmd_yaw = wrap_angle(yaw0 + heading)
        self._assist = assist_level(self.level, self.run) if self._assist_override is None else self._assist_override
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._steps = 0
        self._sums = {k: 0.0 for k in self.STAT_KEYS}

    # -------------------
    # Kinematics
    # -------------------
    def _refresh_kinematics(self) -> None:
        """Brings the derived kinematic fields in line with qpos/qvel (mj_step leaves them one substep behind)."""
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_comVel(self.model, self.data)
        mujoco.mj_subtreeVel(self.model, self.data)

    def _rotation(self) -> np.ndarray:
        """Rotation matrix of the torso (columns = torso axes in the world frame)."""
        return self.data.xmat[self._torso_id].reshape(3, 3)

    def _build_obs(self, R: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        d = self.data
        yaw = np.arctan2(R[1, 0], R[0, 0])
        alpha = wrap_angle(self._cmd_yaw - yaw)
        obs = np.concatenate([
            [d.qpos[2]],
            -R[2, :],                  # gravity direction in the torso frame
            R.T @ com_vel,             # centre-of-mass velocity in the torso frame
            d.qvel[3:6],               # angular velocity, already in the torso frame for a free joint
            d.qpos[7:],
            d.qvel[6:],
            self._prev_action,
            [np.cos(alpha), np.sin(alpha), self._v_target],
        ]).astype(np.float32)
        if self.intensity > 0.0 and self.profile["obs_noise"] > 0.0:
            sigma = self.profile["obs_noise"] * self.intensity * self._run_sigma_unit
            obs += self.rng_dr.standard_normal(obs.shape[0]).astype(np.float32) * sigma
        return obs

    def _get_obs(self) -> np.ndarray:
        self._refresh_kinematics()
        return self._build_obs(self._rotation(), self.data.subtree_linvel[self._torso_id].copy())

    # -------------------
    # Step
    # -------------------
    def _observe_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        p = self.run
        self._refresh_kinematics()
        R = self._rotation()
        com_vel = self.data.subtree_linvel[self._torso_id].copy()

        direction = np.array([np.cos(self._cmd_yaw), np.sin(self._cmd_yaw)])
        v_par = float(com_vel[0] * direction[0] + com_vel[1] * direction[1])
        v_perp = float(-com_vel[0] * direction[1] + com_vel[1] * direction[0])
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        head_err = wrap_angle(self._cmd_yaw - yaw)
        tilt = float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))

        v_target = self._v_target
        sigma = max(0.3, 0.35 * v_target)
        r_vel = float(np.exp(-((v_par - v_target) / sigma) ** 2))
        r_prog = float(np.clip(v_par / max(v_target, 0.1), 0.0, 1.0)) * min(1.0, v_target / 0.5)
        r_head = 0.5 * (1.0 + np.cos(head_err))
        r_up = float(np.exp(-(tilt / 0.5) ** 2))
        ctrl = float(np.sum(np.square(self.data.ctrl)))  # applied torque (fraction of full), not the raw action
        rate = float(np.sum(np.square(action - self._prev_action)))

        reward = (p["w_alive"] + p["w_vel"] * r_vel + p["w_prog"] * r_prog + p["w_head"] * r_head
                  + p["w_up"] * r_up - p["w_lat"] * min(v_perp ** 2, 4.0)
                  - p["w_ctrl"] * ctrl - p["w_rate"] * rate)

        terms = dict(speed=v_par, v_target=v_target, assist=self._assist, dist=v_par * self.dt, lat=abs(v_perp), head_err=abs(head_err),
                     tilt=tilt, r_vel=r_vel, r_prog=r_prog, r_head=r_head, r_up=r_up, ctrl=ctrl, rate=rate)
        self._steps += 1
        for k, v in terms.items():
            self._sums[k] += v

        self._prev_action[:] = action
        done = bool(self.data.qpos[2] < p["fall_height"] or tilt > p["fall_tilt"] or not np.isfinite(reward))
        return self._build_obs(R, com_vel), float(reward), done, {}

    def _is_done(self) -> bool:
        tilt = float(np.arccos(np.clip(self._rotation()[2, 2], -1.0, 1.0)))
        return bool(self.data.qpos[2] < self.run["fall_height"] or tilt > self.run["fall_tilt"])

    # -------------------
    # Episode outcome
    # -------------------
    def episode_stats(self) -> Dict[str, float]:
        """
        Means over the steps of the episode: speed and lateral speed along/across the command, heading
        error, tilt and the reward terms; v_target is the commanded speed and dist the distance covered
        along the commanded direction.
        """
        n = max(self._steps, 1)
        stats = {k: self._sums[k] / n for k in self.STAT_KEYS}
        stats["v_target"] = self._v_target
        stats["dist"] = self._sums["dist"]
        return stats

    def episode_success(self, fell: bool) -> bool:
        if fell or self._steps == 0:
            return False
        mean_speed = self._sums["speed"] / self._steps
        return mean_speed >= self.run["success_speed"] * self._v_target
