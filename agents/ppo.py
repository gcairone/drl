from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from agents.base_agent import BaseAgent
from agents.policy import TorchPolicy, BasePolicy
from models.actor_critic import GaussianActor, Critic

class PPOAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99, clip_ratio: float = 0.2, train_pi_iters: int = 80, train_v_iters: int = 80, target_kl: float = 0.01, device: str = 'cpu') -> None:
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        
        self.actor = GaussianActor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            action_t, log_prob_t = self.actor(state_t, deterministic=not training, with_logprob=True)
            
            value_t = self.critic(state_t)
            
        action = action_t.cpu().numpy()
        log_prob = log_prob_t.cpu().item() if log_prob_t is not None else 0.0
        value = value_t.cpu().item()
            
        return action, {"log_prob": log_prob, "value": value}

    def update(self, buffer: Any) -> Dict[str, float]:
        data = buffer.get()
        obs = data['obs']
        act = data['act']
        ret = data['ret']
        adv = data['adv']
        logp_old = data['logp']

        for i in range(self.train_pi_iters):
            self.optimizer_actor.zero_grad()
            
            dist = self.actor._distribution(obs)
            logp = dist.log_prob(act).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()
            
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
            
            approx_kl = (logp_old - logp).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break
            
            loss_pi.backward()
            self.optimizer_actor.step()

        for i in range(self.train_v_iters):
            self.optimizer_critic.zero_grad()
            value = self.critic(obs).squeeze()
            loss_v = ((value - ret) ** 2).mean()
            loss_v.backward()
            self.optimizer_critic.step()

        return {"loss_actor": loss_pi.item(), "loss_critic": loss_v.item(), "entropy": entropy.item(), "kl": approx_kl}

    def save(self, filepath: str, **kwargs) -> None:
        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'opt_actor': self.optimizer_actor.state_dict(),
            'opt_critic': self.optimizer_critic.state_dict()
        }
        state.update(kwargs)
        torch.save(state, filepath)

    def load(self, filepath: str) -> Dict[str, Any]:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer_actor.load_state_dict(checkpoint['opt_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['opt_critic'])
        return checkpoint

    def get_policy(self) -> BasePolicy:
        # This method is crucial: it extracts only the "Actor" part for use with
        # existing scripts (rollout.py, play.py) that expect a BasePolicy.
        return TorchPolicy(self.actor, device=self.device)
