from typing import Dict, Any, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import numpy as np
from agents.base_agent import BaseAgent
from agents.policy import TorchPolicy, BasePolicy
from models.actor_critic import GaussianActor, Critic
from models.normalizer import RunningObsNormalizer

class PPOAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99, clip_ratio: float = 0.2, update_epochs: int = 10, minibatch_size: int = 64, max_grad_norm: float = 0.5, target_kl: float = 0.01, device: str = 'cpu', obs_norm: bool = False, log_std_min: Optional[float] = None, hidden_layers: Sequence[int] = (64, 64), log_std_init: float = -0.5, adaptive_lr: bool = False, lr_min: float = 1e-5, lr_max: float = 5e-4) -> None:
        """
        Args:
            update_epochs (int): Passes over the collected batch in each update.
            minibatch_size (int): Minibatch size for the SGD steps (<= 0 means full batch).
            max_grad_norm (float): Gradient norm clipping (<= 0 disables it).
            target_kl (float): Stops the update early when the KL to the old policy exceeds
                               1.5 * target_kl after a pass (<= 0 disables it).
            log_std_min (Optional[float]): Floor of the actor's log standard deviation (None: no floor).
            hidden_layers (Sequence[int]): Sizes of the hidden layers of the actor and of the critic.
            log_std_init (float): Initial log standard deviation of the actor.
            adaptive_lr (bool): Adapt the actor's learning rate to the KL of every minibatch (down by 1.5 above
                                2 * target_kl, up below target_kl / 2, within [lr_min, lr_max]). The early stop on the
                                KL is then off: every pass over the batch is done.
        """
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.adaptive_lr = adaptive_lr
        self.lr_min, self.lr_max = lr_min, lr_max
        self.actor_lr = lr
        
        self.log_std_min = log_std_min
        self.hidden_layers = list(hidden_layers)
        self.actor = GaussianActor(state_dim, action_dim, hidden_layers=self.hidden_layers, log_std_init=log_std_init, log_std_min=log_std_min).to(device)
        self.critic = Critic(state_dim, hidden_layers=self.hidden_layers).to(device)
        self.obs_normalizer = None
        if obs_norm:
            self._attach_obs_normalizer()
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Accepts a single state (state_dim,) or a batch (N, state_dim).
        log_prob and value are floats for a single state, arrays of shape (N,) for a batch.
        """
        with torch.inference_mode():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action_t, log_prob_t = self.actor.sample(state_t, deterministic=not training)
            value_t = self.critic(state_t).squeeze(-1)

        action = action_t.cpu().numpy()
        log_prob = log_prob_t.cpu().numpy()
        value = value_t.cpu().numpy()
        if action.ndim == 1:
            log_prob, value = log_prob.item(), value.item()

        return action, {"log_prob": log_prob, "value": value}

    def update(self, buffer: Any) -> Dict[str, float]:
        data = buffer.get()
        obs = data['obs']
        act = data['act']
        ret = data['ret']
        adv = data['adv']
        logp_old = data['logp']

        n = obs.shape[0]
        mb_size = self.minibatch_size if self.minibatch_size > 0 else n
        epochs_done = 0

        for epoch in range(self.update_epochs):
            perm = torch.randperm(n, device=obs.device)
            pi_losses, v_losses = [], []

            for start in range(0, n, mb_size):
                idx = perm[start:start + mb_size]

                dist = self.actor._distribution(obs[idx])
                logp = dist.log_prob(act[idx]).sum(axis=-1)
                ratio = torch.exp(logp - logp_old[idx])
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv[idx]
                loss_pi = -(torch.min(ratio * adv[idx], clip_adv)).mean()

                if self.adaptive_lr:
                    with torch.no_grad():
                        mb_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()  # estimate of KL(old || new)
                    if mb_kl > 2.0 * self.target_kl:
                        self.actor_lr = max(self.actor_lr / 1.5, self.lr_min)
                    elif mb_kl < 0.5 * self.target_kl:
                        self.actor_lr = min(self.actor_lr * 1.5, self.lr_max)
                    for group in self.optimizer_actor.param_groups:
                        group['lr'] = self.actor_lr

                self.optimizer_actor.zero_grad()
                loss_pi.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()
                if self.log_std_min is not None:
                    # keep the stored parameter at the floor too, so it cannot drift far below it
                    with torch.no_grad():
                        self.actor.log_std.clamp_(min=self.log_std_min)

                value = self.critic(obs[idx]).squeeze(-1)
                loss_v = ((value - ret[idx]) ** 2).mean()

                self.optimizer_critic.zero_grad()
                loss_v.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                pi_losses.append(loss_pi.item())
                v_losses.append(loss_v.item())

            epochs_done += 1

            # KL and entropy of the updated policy on the whole batch
            with torch.no_grad():
                dist = self.actor._distribution(obs)
                approx_kl = (logp_old - dist.log_prob(act).sum(axis=-1)).mean().item()
                entropy = dist.entropy().sum(axis=-1).mean().item()
            if self.target_kl > 0 and approx_kl > 1.5 * self.target_kl and not self.adaptive_lr:
                break

        # refresh observation statistics only after the update, so rollout and update used the same ones
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(obs)

        return {"loss_actor": float(np.mean(pi_losses)), "loss_critic": float(np.mean(v_losses)), "entropy": entropy, "kl": approx_kl, "update_epochs": epochs_done, "lr": self.actor_lr}

    def _attach_obs_normalizer(self) -> None:
        self.obs_normalizer = RunningObsNormalizer(self.state_dim).to(self.device)
        self.actor.obs_norm = self.obs_normalizer
        self.critic.obs_norm = self.obs_normalizer

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
        if "obs_norm.mean" in checkpoint['actor'] and self.obs_normalizer is None:
            self._attach_obs_normalizer()
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer_actor.load_state_dict(checkpoint['opt_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['opt_critic'])
        return checkpoint

    def get_policy(self) -> BasePolicy:
        # This method is crucial: it extracts only the "Actor" part for use with
        # existing scripts (rollout.py, play.py) that expect a BasePolicy.
        return TorchPolicy(self.actor, device=self.device)
