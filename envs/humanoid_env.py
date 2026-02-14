from typing import Tuple, Optional, Dict, Any
import os
import mujoco
import mujoco.viewer
import numpy as np

class HumanoidEnv:
    """
    Custom wrapper for MuJoCo Humanoid environment.
    Does not depend on Gymnasium.
    Uses Renderer + RenderContext for controllable step-by-step rendering.
    """

    def __init__(self, model_path: Optional[str] = None, 
                 render: bool = False, 
                 seed: Optional[int] = None, 
                 noise_scale: float = 0.1,
                 ctrl_cost_weight: float = 0.01) -> None:
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

        self._viewer_ctx = None  # for passive viewer

        self.noise_scale = noise_scale
        self.ctrl_cost_weight = ctrl_cost_weight

    # -------------------
    # Simulation Methods
    # -------------------
    def reset(self) -> np.ndarray:
        """
        Resets the environment to the initial state.
        
        Returns:
            np.ndarray: Initial observation.
        """
        mujoco.mj_resetData(self.model, self.data)

        # add slight noise to initial state 
        self.data.qpos[:] += self.rng.uniform(-self.noise_scale, self.noise_scale, size=self.model.nq)
        self.data.qvel[:] += self.rng.uniform(-self.noise_scale, self.noise_scale, size=self.model.nv)

        mujoco.mj_forward(self.model, self.data)
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
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _compute_reward(self) -> float:
        torso_height = self.data.qpos[2]

        alive_reward = torso_height if torso_height > 1.2 else 0.0

        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(self.data.ctrl))

        return alive_reward - ctrl_cost
    
    def _is_done(self) -> bool:
        return self.data.qpos[2] < 0.2  

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