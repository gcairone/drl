# DRL Humanoid

Deep Reinforcement Learning on a Humanoid using MuJoCo

---

## Project Structure

The project is organized into:

*   **`agents/`**: Contains the RL algorithms (e.g., `PPOAgent` in `ppo.py`) and policy interfaces (`policy.py`).
*   **`assets/`**: Stores MuJoCo XML model files (e.g., `humanoid.xml`).
*   **`envs/`**: Custom MuJoCo environment wrappers (`humanoid_env.py`) that handle simulation steps, resets, and rendering.
*   **`models/`**: PyTorch neural network architectures (`actor_critic.py`, `base_dnn.py`).
*   **`scripts/`**: Executable scripts for training, testing, and visualization.
*   **`utils/`**: Helper modules for replay buffers (`buffer.py`) and logging (`logger.py`).

### Output Directories

These directories are generated during usage:

*   **`checkpoints/`**: Stores the trained model weights.
    *   Created by `scripts.train`.
    *   Structure: `checkpoints/<run_id>/model_epoch_<N>.pth`.
    *   Each `.pth` file contains the state dictionaries for the Actor, Critic, and their optimizers.
*   **`data/`**: Contains simulation rollouts (trajectories).
    *   Created by `scripts.rollout`.
    *   Format: `.npz` files (NumPy archives).
    *   Content: Arrays for `qpos`, `qvel`, `actions`, and `rewards` used for faithful replay.
*   **`log/`**: Records training metrics.
    *   Created by `scripts.train`.
    *   Format: CSV files named with the run timestamp (e.g., `YYYY-MM-DD_HH-MM-SS_metrics.csv`).
    *   Content: Epoch stats including Average Return, Loss (Actor/Critic), Entropy, and KL Divergence.

---

## Docker

All dependencies (MuJoCo, CPU-only PyTorch, numpy, pandas, matplotlib) are installed in the container defined by `Dockerfile` / `docker-compose.yml`. The repo is mounted at `/workspace/drl`.

```bash
xhost +local:docker               # only needed for GUI scripts (X11)
docker compose up -d --build
docker compose exec drl bash      # then run the scripts below
```

---

## Test MuJoCo

To check that everything works:

```bash
python3 -m scripts.test_mujoco
```

You should see the Humanoid model loaded correctly.

---

## Rollout and Replay

## Training

```bash
python3 -m scripts.train   
```

* Performs headless RL training
* Saves checkpoints in `models/`
* Logs metrics via `utils/logger.py` in `log/`

---

### Generate trajectories

```bash
python3 -m scripts.rollout
```

This saves trajectories (`qpos` and `qvel`) to a `.npz` file in `data/`

### Play trajectories

```bash
python3 -m scripts.play --file path/to/trajectory.npz --fps 60
```

* Step-by-step replay of saved trajectories
* FPS and skip are configurable

---


