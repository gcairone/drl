import torch
import torch.nn as nn
from agents.base_agent import BaseAgent
from agents.policy import TorchPolicy
from models.base_dnn import BaseDNN

class PPOAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_ratio=0.2, device='cpu'):
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
        # In PPO abbiamo Actor (Policy) e Critic (Value Function)
        # Nota: Per azioni continue, l'Actor dovrà in futuro gestire anche la deviazione standard (std).
        # Per ora usiamo BaseDNN come placeholder per la media delle azioni.
        self.actor = BaseDNN(state_dim, action_dim).to(device)
        self.critic = BaseDNN(state_dim, 1).to(device)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, training=True):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            action_mean = self.actor(state_t)
            # Qui in futuro aggiungerai il sampling: action = normal(mean, std).sample()
            # Per ora ritorniamo la media (comportamento deterministico)
            action = action_mean.cpu().numpy()
            
        # PPO ha bisogno delle log_prob nel buffer, qui ritorneresti anche quelle
        return action, {"log_prob": 0.0, "value": 0.0}

    def update(self, buffer):
        # Qui implementerai il calcolo della loss PPO e l'aggiornamento dei pesi
        # loss = ...
        # optimizer.step()
        return {"loss_actor": 0.0, "loss_critic": 0.0}

    def save(self, filepath):
        # Salviamo tutto ciò che serve per riprendere il training
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'opt_actor': self.optimizer_actor.state_dict(),
            'opt_critic': self.optimizer_critic.state_dict()
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer_actor.load_state_dict(checkpoint['opt_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['opt_critic'])

    def get_policy(self):
        # Questo metodo è cruciale: estrae solo la parte "Actor" per usarla con 
        # gli script esistenti (rollout.py, play.py) che si aspettano una BasePolicy.
        return TorchPolicy(self.actor, device=self.device)