# DRL Humanoid

Personal project to experiment with **Deep Reinforcement Learning (DRL) on a Humanoid** using MuJoCo.  
Includes a custom environment, basic agents, and utilities for training, rollout, and trajectory replay.

---

## Project Structure

```

drl_humanoid/
│
├── assets/
│   └── humanoid.xml           # Standard MuJoCo model
│
├── envs/
│   ├── **init**.py
│   └── humanoid_env.py        # Custom wrapper for MuJoCo Humanoid
│
├── agents/
│   ├── **init**.py
│   ├── policy.py              # BasePolicy and agent classes
│   └── ppo.py                 # PPO implementation
│
├── utils/
│   ├── **init**.py
│   ├── logger.py              # Logging and metrics
│   ├── checkpoint.py          # Model save/load
│   └── config.py              # Experiment configuration
│
├── scripts/
│   ├── train.py               # Main training script
│   ├── rollout.py             # Offline trajectory generation
│   ├── play.py                # Trajectory replay with GUI
│   └── test_mujoco.py         # Basic MuJoCo test
│
├── models/                    # Saved model checkpoints
└── README.md

````

---

## Installation

1. Install **MuJoCo >= 2.3** and the `mjkey.txt` license as described in the [official documentation](https://mujoco.org/).  
2. Create a Python 3.10+ virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

> Note: `requirements.txt` should include at least `mujoco`, `numpy`, and optionally `torch` or other RL libraries.

---

## Test MuJoCo

To check that everything works:

```bash
python3 -m scripts.test_mujoco
```

You should see the Humanoid model loaded correctly.

---

## Rollout and Replay

### Generate trajectories

```bash
python3 -m scripts.rollout
```

This saves trajectories (`qpos` and `qvel`) to a `.npz` file.

### Replay trajectories

```bash
python3 -m scripts.play --file path/to/trajectory.npz --fps 60
```

* Step-by-step replay of saved trajectories
* FPS and skip are configurable
* Fully faithful to the saved trajectory (does not advance physics internally)

---

## Training

```bash
python3 -m scripts.train
```

* Performs headless RL training
* Saves checkpoints in `models/`
* Logs metrics via `utils/logger.py`

---

## Notes

* This is a **personal project** for experimenting with DRL on MuJoCo Humanoids.
* The replay GUI uses `mujoco.viewer.launch_passive()` to ensure trajectory fidelity.
* All training logic is independent from visualization.

---

## References

* [MuJoCo](https://mujoco.org/)
* Personal experiments with Humanoid RL in Python
