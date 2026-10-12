from typing import List, Type, Tuple, Optional, Union
import math
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
        self.mean_net = BaseDNN(state_dim, action_dim, hidden_layers, activation)
        
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        # replaced by a RunningObsNormalizer (shared with the critic) when observation normalization is on
        self.obs_norm: nn.Module = nn.Identity()

    def _distribution(self, state: torch.Tensor) -> Normal:
        """
        Creates the Normal distribution given the state.
        """
        mu = self.mean_net(self.obs_norm(state))
        std = torch.exp(self.log_std)
        return Normal(mu, std, validate_args=False)

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast action sampling for the rollout hot path: avoids building a Normal object per call.
        The log_prob matches _distribution(state).log_prob(action).sum(-1).

        Args:
            state (torch.Tensor): Current state, (state_dim,) or (batch, state_dim).
            deterministic (bool): If True, returns the mean.

        Returns:
            (action, log_prob): Sampled action and its log-probability.
        """
        mu = self.mean_net(self.obs_norm(state))
        noise = torch.zeros_like(mu) if deterministic else torch.randn_like(mu)
        action = mu + torch.exp(self.log_std) * noise
        log_prob = (-0.5 * noise.pow(2) - self.log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        return action, log_prob

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
        self.obs_norm: nn.Module = nn.Identity()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predicts the value of the state V(s).
        """
        return self.v_net(self.obs_norm(state))