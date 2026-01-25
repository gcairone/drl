# envs/humanoid_env_renderer.py
import os
import mujoco
import mujoco.viewer
import numpy as np

class HumanoidEnv:
    """
    Wrapper custom per MuJoCo Humanoid.
    Non dipende da Gym.
    Usa Renderer + RenderContext per rendering passo-passo controllabile.
    """

    def __init__(self, model_path=None, render=False, seed=None):
        # Percorso modello XML
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "../assets/humanoid.xml",
            )
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Humanoid XML file not found: {model_path}")

        # Carica modello e dati
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Dimensioni observation/action
        self.obs_dim = self.model.nq + self.model.nv
        self.action_dim = self.model.nu

        # Random seed
        self.rng = np.random.default_rng(seed)

        self._viewer_ctx = None  # per viewer passive

    # -------------------
    # Metodi simulazione
    # -------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _compute_reward(self):
        return float(self.data.qvel[0])  # velocità torso in avanti

    def _is_done(self):
        return self.data.qpos[2] < 0.8  # humanoid caduto

    def _set_state(self, qpos, qvel):
        """Imposta qpos e qvel senza fare simulazione fisica aggiuntiva."""
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)


    def render_init_passive(self):
        """
        Inizializza il viewer in modalità passive (non-blocking) per replay fedele.
        Restituisce il viewer handle.
        """
        if self._viewer_ctx is None:
            self._viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer_ctx

    def render_sync(self, state_only=True):
        """
        Aggiorna il viewer. 
        state_only=True -> solo stato, senza avanzare la fisica.
        """
        if self._viewer_ctx is not None and self._viewer_ctx.is_running():
            self._viewer_ctx.sync(state_only=state_only)

    def render_close(self):
        """
        Chiude il viewer.
        """
        if self._viewer_ctx is not None:
            self._viewer_ctx.close()
            self._viewer_ctx = None