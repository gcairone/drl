from typing import Union
import numpy as np
import torch

class BasePolicy:
    """
    Base class for policies.
    """

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Select an action based on the given observation.

        Args:
            observation (np.ndarray): The current observation from the environment.

        Returns:
            np.ndarray: The action to be taken.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return self.act(observation)
    
class NullPolicy(BasePolicy):
    """
    Policy that returns null actions.
    Useful for testing the passive behavior of the environment.
    """
    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_dim)
    
class RandomPolicy(BasePolicy):
    """
    Policy that returns random actions within specified bounds.
    """
    def __init__(self, action_dim: int, low: float = -1.0, high: float = 1.0) -> None:
        self.action_dim = action_dim
        self.low = low
        self.high = high

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=self.action_dim)
    
class TorchPolicy(BasePolicy):
    """
    Wrapper to use a PyTorch model as a policy.
    """
    def __init__(self, model: torch.nn.Module, device: str = 'cpu') -> None:   
        self.model = model
        self.device = device

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_tensor = self.model(obs_tensor)
        return action_tensor.cpu().numpy()