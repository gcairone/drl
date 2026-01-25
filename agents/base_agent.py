import torch
import numpy as np

class BaseAgent:
    """
    Classe base astratta per agenti RL (PPO, SAC, DQN, etc.).
    Definisce l'interfaccia comune per training, salvataggio e selezione azioni.
    """
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    def select_action(self, state, training=True):
        """
        Seleziona un'azione dato uno stato.
        
        Args:
            state (np.ndarray): Stato corrente.
            training (bool): Se True, usa esplorazione (es. sampling stocastico).
                             Se False, usa exploitazione (es. media deterministica).
        
        Returns:
            action (np.ndarray): Azione da applicare nell'ambiente.
            extra_info (dict): Info aggiuntive (es. log_prob per PPO, Q-val per DQN) utili per il buffer.
        """
        raise NotImplementedError

    def update(self, buffer):
        """
        Aggiorna i parametri dell'agente usando i dati raccolti nel buffer.
        
        Returns:
            metrics (dict): Dizionario con metriche di training (loss, entropy, etc.) per il logger.
        """
        raise NotImplementedError

    def save(self, filepath):
        """Salva lo stato completo dell'agente (pesi reti + stati ottimizzatori)."""
        raise NotImplementedError

    def load(self, filepath):
        """Carica lo stato completo dell'agente."""
        raise NotImplementedError

    def get_policy(self):
        """
        Restituisce un oggetto Policy (compatibile con agents/policy.py) 
        per l'uso in rollout/play separati dal training.
        Di solito restituisce l'Actor wrappato in una TorchPolicy.
        """
        raise NotImplementedError