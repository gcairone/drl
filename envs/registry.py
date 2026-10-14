"""Registry of the tasks: name -> environment class and its task-specific parameters with their defaults."""
from typing import Dict, Type

from envs.humanoid_env import HumanoidEnv
from envs.run_env import HumanoidRunEnv, RUN_DEFAULTS

TASKS: Dict[str, Type[HumanoidEnv]] = {"stand": HumanoidEnv, "run": HumanoidRunEnv}
TASK_PARAMS: Dict[str, Dict[str, float]] = {"stand": {}, "run": RUN_DEFAULTS}
DEFAULT_HIDDEN: Dict[str, tuple] = {"stand": (64, 64), "run": (256, 256)}


def env_class(task: str) -> Type[HumanoidEnv]:
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}', choose one of {sorted(TASKS)}")
    return TASKS[task]

# (initial log std, floor of the log std) of the policy
DEFAULT_LOG_STD: Dict[str, tuple] = {"stand": (-0.5, -1.2), "run": (-1.0, -1.6)}

# step of the curriculum level per decision: the running task needs a slower one, its assist fades 4x faster than the level
DEFAULT_CURR_STEP: Dict[str, float] = {"stand": 0.02, "run": 0.005}
