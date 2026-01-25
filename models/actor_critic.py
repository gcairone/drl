from typing import List, Type, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from models.base_dnn import BaseDNN

class GaussianActor(nn.Module):
    """
    Gaussian Actor for continuous action spaces.
    Predicts mean and standard deviation for a Normal distribution.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int] = [64, 64], activation: Type[nn.Module] = nn.ReLU, log_std_init: float = -0.5) -> None:
        super().__init__()
        # The mean is predicted by the neural network
        self.mean_net = BaseDNN(state_dim, action_dim, hidden_layers, activation)
        
        # The log_std is a learnable parameter, independent of the state (common in PPO)
        # Initialized to a specific value (e.g. -0.5 -> std approx 0.6)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def _distribution(self, state: torch.Tensor) -> Normal:
        """
        Creates the Normal distribution given the state.
        """
        mu = self.mean_net(state)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, state: torch.Tensor, deterministic: bool = True, with_logprob: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass to select an action.

        Args:
            state (torch.Tensor): Current state.
            deterministic (bool): If True, returns the mean. If False, samples from distribution. Defaults to True.
            with_logprob (bool): If True, returns (action, log_prob). If False, returns only action. Defaults to False.

        Returns:
            action (torch.Tensor) or (action, log_prob): The selected action, and optionally log_prob.
        """
        dist = self._distribution(state)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        if with_logprob:
            # Sum over the action dimension to get the log_prob of the entire action vector
            log_prob = dist.log_prob(action).sum(axis=-1)
            return action, log_prob
        
        return action

class Critic(nn.Module):
    """
    Critic network (Value Function) estimating V(s).
    """
    def __init__(self, state_dim: int, hidden_layers: List[int] = [64, 64], activation: Type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.v_net = BaseDNN(state_dim, 1, hidden_layers, activation)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predicts the value of the state V(s).
        """
        return self.v_net(state)