from typing import Optional, Tuple, Dict, Any, List, Type
import numpy as np
from envs.humanoid_env import HumanoidEnv

class VecHumanoidEnv:
    """
    Runs N independent HumanoidEnv instances in the same process, stepped in lockstep.
    The policy acts on the whole batch of observations with a single forward pass,
    which amortizes the per-call PyTorch overhead.
    No auto-reset: the caller resets finished envs with reset_at(i), so it can
    bootstrap on the last observation first.
    """

    def __init__(self, num_envs: int, seed: Optional[int] = None, env_cls: Type[HumanoidEnv] = HumanoidEnv,
                 **env_kwargs: Any) -> None:
        self.envs = [
            env_cls(seed=None if seed is None else seed + i, **env_kwargs)
            for i in range(num_envs)
        ]
        self.num_envs = num_envs
        self.obs_dim = self.envs[0].obs_dim
        self.action_dim = self.envs[0].action_dim

    def reset(self) -> np.ndarray:
        """
        Resets all environments.

        Returns:
            np.ndarray: Observations, shape (num_envs, obs_dim).
        """
        return np.stack([env.reset() for env in self.envs])

    def set_difficulty(self, difficulty: float, random_amplitude: bool = True) -> None:
        """
        Sets the randomization level of every environment (see HumanoidEnv.set_difficulty).
        It takes effect from each environment's next reset.
        """
        for env in self.envs:
            env.set_difficulty(difficulty, random_amplitude)

    def reset_at(self, i: int) -> np.ndarray:
        """
        Resets environment i.

        Returns:
            np.ndarray: Its initial observation, shape (obs_dim,).
        """
        return self.envs[i].reset()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Advances every environment by one step.

        Args:
            actions (np.ndarray): Actions, shape (num_envs, action_dim).

        Returns:
            Tuple: Observations (num_envs, obs_dim), rewards (num_envs,), done flags (num_envs,), info dicts.
        """
        obs, rews, dones, infos = zip(*[env.step(a) for env, a in zip(self.envs, actions)])
        return np.stack(obs), np.array(rews, dtype=np.float32), np.array(dones, dtype=bool), list(infos)
