import numpy as np
import torch
from typing import Dict

class PPOBuffer:
    """
    Buffer for storing trajectories and calculating GAE (Generalized Advantage Estimation) for PPO.
    Stores transitions (obs, act, rew, val, logp) and computes advantages at the end of trajectories.
    """
    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float = 0.99, lam: float = 0.95, device: str = 'cpu') -> None:
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        self.device = device

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float) -> None:
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size, "Buffer overflow!"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0) -> None:
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        Args:
            last_val (float): The value of the last state encountered. 
                              Should be 0 if the trajectory ended because of a terminal state (done=True),
                              otherwise V(s_T) if it was cut off by max_steps.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-Lambda calculation
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        
        # Calculate advantages recursively
        # A_t = delta_t + (gamma * lambda) * A_{t+1}
        advs = np.zeros_like(deltas)
        last_gae_lam = 0
        for t in reversed(range(len(deltas))):
            last_gae_lam = deltas[t] + self.gamma * self.lam * last_gae_lam
            advs[t] = last_gae_lam
            
        self.adv_buf[path_slice] = advs
        
        # The targets for the value function are the "returns" (rewards-to-go)
        # Return = Advantage + Value
        self.ret_buf[path_slice] = advs + vals[:-1]
        
        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size, "Buffer not full before get()!"
        self.ptr, self.path_start_idx = 0, 0
        
        # Advantage Normalization (Standardization)
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            obs=self.obs_buf, 
            act=self.act_buf, 
            ret=self.ret_buf,
            adv=self.adv_buf, 
            logp=self.logp_buf,
            val=self.val_buf
        )
        
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in data.items()}