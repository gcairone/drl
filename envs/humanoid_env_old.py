# envs/humanoid_env.py
import os
import mujoco
import mujoco.viewer
import numpy as np


class HumanoidEnv:
    """
    Wrapper custom per MuJoCo Humanoid.
    Non dipende da Gym.
    Usa il viewer ufficiale mujoco.viewer.
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

        # Rendering
        self.render_mode = render
        self._viewer_ctx = None

        # Dimensioni observation/action
        self.obs_dim = self.model.nq + self.model.nv
        self.action_dim = self.model.nu

        # Random seed
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """
        Resetta l'ambiente e ritorna l'osservazione iniziale.
        """
        mujoco.mj_resetData(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        """
        Applica action al simulatore e ritorna obs, reward, done, info.
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

    def render(self):
        """
        Apre il viewer MuJoCo (blocking).
        """
        if not self.render_mode:
            return

        if self._viewer_ctx is None:
            # Launch blocking viewer so caller (script) waits until the GUI is closed.
            mujoco.viewer.launch(self.model, self.data)

    def close(self):
        """
        Chiude il viewer.
        """
        if self._viewer_ctx is not None:
            self._viewer_ctx.close()
            self._viewer_ctx = None

    def _get_obs(self):
        """
        Osservazione: concatenazione qpos + qvel.
        """
        return np.concatenate(
            [self.data.qpos, self.data.qvel]
        ).astype(np.float32)

    def _compute_reward(self):
        """
        Reward semplice: velocità in avanti del torso.
        """
        # Assumiamo asse x come direzione di avanzamento
        forward_velocity = self.data.qvel[0]
        return float(forward_velocity)

    def _is_done(self):
        """
        Termina episodio se humanoid cade.
        """
        z_pos = self.data.qpos[2]  # altezza torso
        return z_pos < 0.8

    def _set_state(self, qpos, qvel):
        """Imposta qpos e qvel, e aggiorna stato interno."""
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def render_step(self):
        """
        Renderizza un singolo step.
        Se render_mode=False, non fa nulla.
        """
        if not self.render_mode:
            return
        if self._viewer_ctx is None:
            self._viewer_ctx = mujoco.viewer.launch(self.model, self.data)
        else:
            self._viewer_ctx.render()