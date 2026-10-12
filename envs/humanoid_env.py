from typing import Tuple, Optional, Dict, Any
import os
import mujoco
import mujoco.viewer
import numpy as np

# Randomization profile. Every value is the maximum amplitude reached at difficulty 1;
# amplitudes scale linearly with the intensity of the episode (see HumanoidEnv.set_difficulty).
# At difficulty 0 the environment is the plain one: only `noise_scale` is applied at reset.
RANDOMIZATION_DEFAULTS: Dict[str, float] = dict(
    # --- reset randomization
    rand_tilt=0.15,       # rad: random root tilt (roll/pitch)
    rand_joint_pos=0.10,  # rad: random joint angle offsets (clipped to the joint ranges)
    rand_vel=0.30,        # m/s, rad/s: random initial velocities of every dof
    # --- domain randomization, drawn at every reset
    dr_mass=0.50,         # relative: each link mass (and inertia) scaled by U(1-x, 1+x)
    dr_friction=0.70,     # relative: friction of all geoms scaled by one factor U(1-x, 1+x)
    dr_gear=0.50,         # relative: each actuator strength scaled by U(1-x, 1+x)
    dr_damping=1.00,      # relative: each joint damping scaled by U(1-x, 1+x)
    obs_noise=0.02,       # std of the Gaussian observation noise on positions (velocities: x10)
    # --- external disturbances during the episode
    push_force=150.0,     # N: peak horizontal force on the torso
    push_interval=3.0,    # s: mean time between pushes (exponentially distributed)
    push_duration=0.10,   # s: duration of each push
)


class HumanoidEnv:
    """
    Custom wrapper for MuJoCo Humanoid environment.
    Does not depend on Gymnasium.
    Uses Renderer + RenderContext for controllable step-by-step rendering.
    """
    MAX_SELF_PENETRATION = 0.01  # m: deepest limb-limb overlap accepted at the start of an episode

    def __init__(self, model_path: Optional[str] = None,
                 render: bool = False,
                 seed: Optional[int] = None,
                 noise_scale: float = 0.01,
                 ctrl_cost_weight: float = 0.01,
                 min_height: float = 1.0,
                 term_height: float = 1.0,
                 frame_skip: int = 3,
                 difficulty: float = 0.0,
                 random_amplitude: bool = True,
                 **profile: float) -> None:
        """
        Args:
            noise_scale (float): Uniform noise added to qpos/qvel at reset. It also moves the torso
                                 height, so large values start the feet inside the floor.
            min_height (float): Torso height above which the alive reward is given.
            term_height (float): Torso height below which the episode terminates.
            frame_skip (int): Physics steps per action (control period = frame_skip * timestep).
            difficulty (float): Randomization level in [0, 1], see set_difficulty().
            random_amplitude (bool): See set_difficulty().
            **profile (float): Overrides of RANDOMIZATION_DEFAULTS (maximum amplitudes at difficulty 1).
        """
        unknown = set(profile) - set(RANDOMIZATION_DEFAULTS)
        if unknown:
            raise TypeError(f"Unknown randomization parameters: {sorted(unknown)}")
        self.profile: Dict[str, float] = {**RANDOMIZATION_DEFAULTS, **profile}

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "../assets/humanoid.xml",
            )
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Humanoid XML file not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.obs_dim = self.model.nq + self.model.nv
        self.action_dim = self.model.nu

        self.rng = np.random.default_rng(seed)
        # separate stream for the randomization: the draws of the plain reset stay untouched
        self.rng_dr = np.random.default_rng(None if seed is None else seed + 7919)

        self._viewer_ctx = None  # for passive viewer

        self.noise_scale = noise_scale
        self.ctrl_cost_weight = ctrl_cost_weight
        self.min_height = min_height
        self.term_height = term_height
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip

        # randomization state
        m = self.model
        assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "expected a free root joint"
        self._nominal = dict(
            body_mass=m.body_mass.copy(),
            body_inertia=m.body_inertia.copy(),
            geom_friction=m.geom_friction.copy(),
            actuator_gear=m.actuator_gear.copy(),
            dof_damping=m.dof_damping.copy(),
        )
        self._dr_active = False
        self._torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._hinge_qpos = m.jnt_qposadr[1:].copy()   # joints after the free root joint
        self._hinge_range = m.jnt_range[1:].copy()
        self._obs_sigma_unit = np.ones(self.obs_dim, dtype=np.float32)
        self._obs_sigma_unit[m.nq:] = 10.0
        self._push_enabled = False
        self._push_left = 0
        self._push_wait = 0
        self.intensity = 0.0
        self.set_difficulty(difficulty, random_amplitude)

    # -------------------
    # Randomization
    # -------------------
    def set_difficulty(self, difficulty: float, random_amplitude: bool = True) -> None:
        """
        Sets the randomization level in [0, 1]; it takes effect from the next reset().
        The episode intensity multiplies every amplitude of the profile: with random_amplitude
        each episode draws it as difficulty * U(0, 1), so the agent sees a mix of levels up to
        the current one; otherwise it is exactly `difficulty` (use this for evaluation).
        Difficulty 0 disables everything beyond `noise_scale`.
        """
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.random_amplitude = random_amplitude

    def _restore_nominal(self) -> bool:
        """Puts the nominal physical parameters back. Returns True if they had been changed."""
        if not self._dr_active:
            return False
        m, nom = self.model, self._nominal
        m.body_mass[:] = nom["body_mass"]
        m.body_inertia[:] = nom["body_inertia"]
        m.geom_friction[:] = nom["geom_friction"]
        m.actuator_gear[:] = nom["actuator_gear"]
        m.dof_damping[:] = nom["dof_damping"]
        self._dr_active = False
        return True

    def _randomize_dynamics(self, s: float) -> None:
        """Domain randomization of the physical parameters, scaled by the episode intensity s."""
        p, r, m, nom = self.profile, self.rng_dr, self.model, self._nominal

        def factor(amplitude: float, size: int) -> np.ndarray:
            return 1.0 + r.uniform(-1.0, 1.0, size=size) * amplitude * s

        if p["dr_mass"] > 0:
            k = factor(p["dr_mass"], m.nbody)
            m.body_mass[:] = nom["body_mass"] * k
            m.body_inertia[:] = nom["body_inertia"] * k[:, None]
            self._dr_active = True
        if p["dr_friction"] > 0:
            m.geom_friction[:, 0] = nom["geom_friction"][:, 0] * factor(p["dr_friction"], 1)
            self._dr_active = True
        if p["dr_gear"] > 0:
            m.actuator_gear[:, 0] = nom["actuator_gear"][:, 0] * factor(p["dr_gear"], m.nu)
            self._dr_active = True
        if p["dr_damping"] > 0:
            m.dof_damping[:] = nom["dof_damping"] * factor(p["dr_damping"], m.nv)
            self._dr_active = True

    def _randomize_state(self, s: float) -> None:
        """Random initial tilt, joint angles and velocities, scaled by the episode intensity s."""
        p, r = self.profile, self.rng_dr
        qpos, qvel = self.data.qpos, self.data.qvel

        if p["rand_tilt"] > 0:
            # small random rotation about the horizontal axes, composed with the current orientation
            rotvec = np.array([r.uniform(-1, 1), r.uniform(-1, 1), 0.0]) * p["rand_tilt"] * s
            quat = qpos[3:7].copy()
            mujoco.mju_quatIntegrate(quat, rotvec, 1.0)
            qpos[3:7] = quat / np.linalg.norm(quat)
        if p["rand_joint_pos"] > 0:
            # redraw the joint offsets until the limbs do not start inside each other
            base = qpos[self._hinge_qpos].copy()
            lo, hi = self._hinge_range.T
            for attempt in range(6):
                noise = r.uniform(-1, 1, size=len(self._hinge_qpos)) * p["rand_joint_pos"] * s
                if attempt == 5:
                    noise[:] = 0.0  # last resort: keep the nominal posture
                qpos[self._hinge_qpos] = np.clip(base + noise, lo, hi)
                mujoco.mj_forward(self.model, self.data)
                if self._self_penetration() <= self.MAX_SELF_PENETRATION:
                    break
        if p["rand_vel"] > 0:
            qvel[:] += r.uniform(-1, 1, size=self.model.nv) * p["rand_vel"] * s

    def _self_penetration(self) -> float:
        """Deepest penetration (m) between two body geoms, floor contacts excluded."""
        n = self.data.ncon
        if n == 0:
            return 0.0
        contact = self.data.contact
        body = (contact.geom1[:n] != self._floor_geom) & (contact.geom2[:n] != self._floor_geom)
        return float(-contact.dist[:n][body].min()) if body.any() else 0.0

    def _lift_out_of_floor(self) -> None:
        """Raises the root until no geom starts inside the floor (the random tilt can cause it)."""
        for _ in range(20):
            n = self.data.ncon
            if n == 0:
                return
            contact = self.data.contact
            on_floor = (contact.geom1[:n] == self._floor_geom) | (contact.geom2[:n] == self._floor_geom)
            if not on_floor.any():
                return
            penetration = -contact.dist[:n][on_floor].min()
            if penetration <= 1e-4:
                return
            self.data.qpos[2] += penetration + 1e-3
            mujoco.mj_forward(self.model, self.data)

    def _draw_push_wait(self) -> int:
        return int(self.rng_dr.exponential(self.profile["push_interval"] / self.dt)) + 1

    def _start_push(self) -> None:
        r = self.rng_dr
        angle = r.uniform(0.0, 2.0 * np.pi)
        magnitude = self.profile["push_force"] * self.intensity * r.uniform(0.3, 1.0)
        self.data.xfrc_applied[self._torso_id, :3] = magnitude * np.array([np.cos(angle), np.sin(angle), 0.0])
        self._push_left = max(1, int(round(self.profile["push_duration"] / self.dt)))

    def _update_push(self) -> None:
        """Called once per control step, before the physics: starts and stops the random pushes."""
        if not self._push_enabled:
            return
        if self._push_left > 0:
            self._push_left -= 1
            if self._push_left == 0:
                self.data.xfrc_applied[self._torso_id, :3] = 0.0
                self._push_wait = self._draw_push_wait()
        else:
            self._push_wait -= 1
            if self._push_wait <= 0:
                self._start_push()

    # -------------------
    # Simulation Methods
    # -------------------
    def reset(self) -> np.ndarray:
        """
        Resets the environment to the initial state.

        Returns:
            np.ndarray: Initial observation.
        """
        restored = self._restore_nominal()

        # intensity of this episode: scales every randomization amplitude
        if self.difficulty > 0.0:
            u = self.rng_dr.uniform() if self.random_amplitude else 1.0
            self.intensity = self.difficulty * u
        else:
            self.intensity = 0.0
        s = self.intensity

        if s > 0.0:
            self._randomize_dynamics(s)
        if restored or self._dr_active:
            # mass changes alter derived constants (e.g. the inverse weights used by the contact solver)
            mujoco.mj_setConst(self.model, self.data)

        mujoco.mj_resetData(self.model, self.data)  # also clears xfrc_applied

        # add slight noise to initial state
        self.data.qpos[:] += self.rng.uniform(-self.noise_scale, self.noise_scale, size=self.model.nq)
        self.data.qvel[:] += self.rng.uniform(-self.noise_scale, self.noise_scale, size=self.model.nv)
        if s > 0.0:
            self._randomize_state(s)

        mujoco.mj_forward(self.model, self.data)
        if s > 0.0:
            self._lift_out_of_floor()

        self._push_enabled = s > 0.0 and self.profile["push_force"] > 0.0
        self._push_left = 0
        if self._push_enabled:
            self._push_wait = self._draw_push_wait()
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Advances the simulation by one step.

        Args:
            action (np.ndarray): Action to apply.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation, reward, done flag, info dict.
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        self._update_push()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        if self.intensity > 0.0 and self.profile["obs_noise"] > 0.0:
            sigma = self.profile["obs_noise"] * self.intensity * self._obs_sigma_unit
            obs += self.rng_dr.standard_normal(self.obs_dim).astype(np.float32) * sigma
        return obs

    def _compute_reward(self) -> float:
        torso_height = self.data.qpos[2]

        alive_reward = torso_height if torso_height > self.min_height else 0.0

        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(self.data.ctrl))

        return alive_reward - ctrl_cost

    def _is_done(self) -> bool:
        return self.data.qpos[2] < self.term_height

    def _set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """
        Sets qpos and qvel without advancing the physical simulation.
        """
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)


    def render_init_passive(self) -> Any:
        """
        Initializes the viewer in passive mode (non-blocking) for faithful replay.

        Returns:
            Any: The viewer handle.
        """
        if self._viewer_ctx is None:
            self._viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer_ctx

    def render_sync(self, state_only: bool = True) -> None:
        """
        Updates the viewer.

        Args:
            state_only (bool): If True, updates only the state without advancing physics.
        """
        if self._viewer_ctx is not None and self._viewer_ctx.is_running():
            self._viewer_ctx.sync(state_only=state_only)

    def render_close(self) -> None:
        """
        Closes the viewer.
        """
        if self._viewer_ctx is not None:
            self._viewer_ctx.close()
            self._viewer_ctx = None
