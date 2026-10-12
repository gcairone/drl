import numpy as np
import torch
from typing import Dict, Any

class PPOBuffer:
    """
    Buffer for storing trajectories and calculating GAE (Generalized Advantage Estimation) for PPO.
    Stores transitions (obs, act, rew, val, logp) and computes advantages at the end of trajectories.
    Supports N parallel environments: data is stored as (steps, num_envs, ...) and each env
    keeps its own trajectory start index.
    """
    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float = 0.99, lam: float = 0.95, device: str = 'cpu', num_envs: int = 1) -> None:
        """
        Args:
            size (int): Total transitions per epoch, split evenly across the environments.
            num_envs (int): Number of parallel environments (must divide size).
        """
        assert size % num_envs == 0, "size must be divisible by num_envs"
        steps = size // num_envs
        self.obs_buf = np.zeros((steps, num_envs, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((steps, num_envs, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros((steps, num_envs), dtype=np.float32)
        self.rew_buf = np.zeros((steps, num_envs), dtype=np.float32)
        self.ret_buf = np.zeros((steps, num_envs), dtype=np.float32)
        self.val_buf = np.zeros((steps, num_envs), dtype=np.float32)
        self.logp_buf = np.zeros((steps, num_envs), dtype=np.float32)
        
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = np.zeros(num_envs, dtype=np.int64)
        self.max_size = steps
        self.num_envs = num_envs
        self.device = device

    def store(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, val: np.ndarray, logp: np.ndarray) -> None:
        """
        Append one timestep of agent-environment interaction (all environments) to the buffer.
        Inputs have a leading num_envs dimension; with num_envs=1 unbatched inputs also work.
        """
        assert self.ptr < self.max_size, "Buffer overflow!"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0, env_idx: int = 0) -> None:
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
            env_idx (int): Index of the environment whose trajectory ended.
        """
        path_slice = slice(self.path_start_idx[env_idx], self.ptr)
        rews = np.append(self.rew_buf[path_slice, env_idx], last_val)
        vals = np.append(self.val_buf[path_slice, env_idx], last_val)
        
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
            
        self.adv_buf[path_slice, env_idx] = advs
        
        self.ret_buf[path_slice, env_idx] = advs + vals[:-1]
        
        self.path_start_idx[env_idx] = self.ptr

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        The environment dimension is flattened: tensors have steps * num_envs rows.
        """
        assert self.ptr == self.max_size, "Buffer not full before get()!"
        self.ptr = 0
        self.path_start_idx[:] = 0
        
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            obs=self.obs_buf.reshape(-1, self.obs_buf.shape[-1]), 
            act=self.act_buf.reshape(-1, self.act_buf.shape[-1]), 
            ret=self.ret_buf.reshape(-1),
            adv=self.adv_buf.reshape(-1), 
            logp=self.logp_buf.reshape(-1),
            val=self.val_buf.reshape(-1)
        )
        
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in data.items()}
    
    def filestore(self, filename: str) -> None:
        """
        Save the buffer contents to a file.
        
        Args:
            filename (str): The file path to save the buffer data.
        """
        np.savez_compressed(
            filename,
            obs=self.obs_buf,
            act=self.act_buf,
            adv=self.adv_buf,
            rew=self.rew_buf,
            ret=self.ret_buf,
            val=self.val_buf,
            logp=self.logp_buf
        )
        # print(f"Buffer saved to {filename}")
        
    def fileread(self, filename: str) -> None:
        """
        Load the buffer contents from a file.
        
        Args:
            filename (str): The file path to load the buffer data from.
        """
        data = np.load(filename)
        self.obs_buf = data['obs']
        self.act_buf = data['act']
        self.adv_buf = data['adv']
        self.rew_buf = data['rew']
        self.ret_buf = data['ret']
        self.val_buf = data['val']
        self.logp_buf = data['logp']
        self.ptr = self.max_size  # Assume buffer is full after loading
        # print(f"Buffer loaded from {filename}")

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the buffer as a dictionary."""
        return {
            'obs_buf': self.obs_buf,
            'act_buf': self.act_buf,
            'adv_buf': self.adv_buf,
            'rew_buf': self.rew_buf,
            'ret_buf': self.ret_buf,
            'val_buf': self.val_buf,
            'logp_buf': self.logp_buf,
            'ptr': self.ptr,
            'path_start_idx': self.path_start_idx
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the buffer state from a dictionary."""
        self.obs_buf = state_dict['obs_buf']
        self.act_buf = state_dict['act_buf']
        self.adv_buf = state_dict['adv_buf']
        self.rew_buf = state_dict['rew_buf']
        self.ret_buf = state_dict['ret_buf']
        self.val_buf = state_dict['val_buf']
        self.logp_buf = state_dict['logp_buf']
        self.ptr = state_dict['ptr']
        self.path_start_idx = state_dict['path_start_idx']