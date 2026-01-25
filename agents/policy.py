import numpy as np

class BasePolicy:
    """
    Base class for policies.
    """

    def act(self, observation):
        """
        Select an action based on the given observation.

        Args:
            observation (np.ndarray): The current observation from the environment.

        Returns:
            np.ndarray: The action to be taken.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def __call__(self, observation):
        return self.act(observation)
    
class NullPolicy(BasePolicy):
    """
    Policy that returns null actions.
    Useful for testing the passive behavior of the environment.
    """
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def act(self, observation):
        return np.zeros(self.action_dim)
    
class RandomPolicy(BasePolicy):
    """
    Policy that returns random actions within specified bounds.
    """
    def __init__(self, action_dim, low=-1.0, high=1.0):
        self.action_dim = action_dim
        self.low = low
        self.high = high

    def act(self, observation):
        return np.random.uniform(self.low, self.high, size=self.action_dim)
    
class TorchPolicy(BasePolicy):
    """
    Wrapper to use a PyTorch model as a policy.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def act(self, observation):
        import torch
        obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_tensor = self.model(obs_tensor)
        return action_tensor.cpu().numpy()