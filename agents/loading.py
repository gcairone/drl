from typing import Any, Dict, Tuple

import torch

from agents.ppo import PPOAgent
from envs.registry import env_class, DEFAULT_HIDDEN


def load_agent(path: str) -> Tuple[PPOAgent, Dict[str, Any], Dict[str, Any]]:
    """
    Loads a PPO checkpoint in eval mode, rebuilding the network it was trained with.
    Checkpoints without an 'arch' entry come from the standing task with the default network.

    Returns:
        (agent, checkpoint dict, env_kwargs the checkpoint was trained with)
    """
    arch = torch.load(path, map_location="cpu", weights_only=False).get("arch", {})
    task = arch.get("task", "stand")
    probe = env_class(task)(render=False)
    agent = PPOAgent(state_dim=probe.obs_dim, action_dim=probe.action_dim, device="cpu",
                     hidden_layers=arch.get("hidden_layers", DEFAULT_HIDDEN["stand"]))
    ckpt = agent.load(path)
    agent.actor.eval()
    agent.critic.eval()
    return agent, ckpt, ckpt.get("env_kwargs", {})


def task_of(ckpt: Dict[str, Any]) -> str:
    return ckpt.get("arch", {}).get("task", "stand")
