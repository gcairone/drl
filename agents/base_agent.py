from typing import Dict, Any, Tuple
import torch
import numpy as np
from agents.policy import BasePolicy

class BaseAgent:
    """
    Abstract base class for RL agents (PPO, SAC, DQN, etc.).
    Defines the common interface for training, saving, and action selection.
    """
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu') -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Selects an action given a state.
        
        Args:
            state (np.ndarray): Current state.
            training (bool): If True, uses exploration (e.g., stochastic sampling).
                             If False, uses exploitation (e.g., deterministic mean).
        
        Returns:
            action (np.ndarray): Action to apply in the environment.
            extra_info (dict): Additional info (e.g., log_prob for PPO, Q-val for DQN) useful for the buffer.
        """
        raise NotImplementedError

    def update(self, buffer: Any) -> Dict[str, float]:
        """
        Updates the agent's parameters using data collected in the buffer.
        
        Args:
            buffer (Any): The replay buffer containing training data.
        
        Returns:
            metrics (dict): Dictionary with training metrics (loss, entropy, etc.) for the logger.
        """
        raise NotImplementedError

    def save(self, filepath: str) -> None:
        """Saves the complete agent state (network weights + optimizer states)."""
        raise NotImplementedError

    def load(self, filepath: str) -> None:
        """Loads the complete agent state."""
        raise NotImplementedError

    def get_policy(self) -> BasePolicy:
        """
        Returns a Policy object (compatible with agents/policy.py) 
        for use in rollout/play separate from training.
        Usually returns the Actor wrapped in a TorchPolicy.
        
        Returns:
            BasePolicy: The policy object.
        """
        raise NotImplementedError