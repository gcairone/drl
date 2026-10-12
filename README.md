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

### Checkpoint policy

Training checkpoints (`results/`, `checkpoints/`) are **not versioned** (see `.gitignore`). Only the best one is, in `pretrained/`:

*   `pretrained/best.pth`: the best checkpoint so far (full PPO checkpoint: usable with `rollout`, `eval_policy` and `train --resume`).
*   `pretrained/best.json`: where it came from and its metrics.

To submit a new checkpoint for promotion:

```bash
python3 -m scripts.promote_best --candidate results/checkpoints/model_epoch_N.pth
```

It evaluates the candidate **and** the current best under the same fixed protocol (100 episodes, deterministic actions, reset noise 0.01, cap 9000 steps = 135 s, seed 12345) and replaces `best.pth` only if the candidate has a strictly higher mean standing time. Then commit `pretrained/`.

To evaluate checkpoints in more detail (stochastic mode, robustness to reset noise, survival rate):

```bash
python3 -m scripts.eval_policy --model pretrained/best.pth --episodes 100
```

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

### Robustness training

By default training uses an **adaptive curriculum** over a single *difficulty* level in [0, 1] (`--no-curriculum --difficulty D` keeps it fixed). At difficulty 0 the environment is the plain one. The level rises by `--curr_step` when at least `--curr_up` of the recent episodes survive the whole time limit, falls when fewer than `--curr_down` do, and is stored in the checkpoints (`--resume` continues from it). Each episode draws its own intensity `difficulty * U(0, 1)`, so the policy keeps seeing easy and hard cases. The intensity scales every amplitude below (maximum values at difficulty 1, set with the option of the same name):

| Group | Option | Max amplitude |
|---|---|---|
| Reset | `--rand_tilt` | 0.15 rad root tilt (roll/pitch) |
| | `--rand_joint_pos` | 0.10 rad joint offsets (redrawn if limbs would start inside each other) |
| | `--rand_vel` | 0.30 initial velocities |
| Domain randomization (per episode) | `--dr_mass` / `--dr_gear` | ±50% link masses (with inertia) / actuator strengths |
| | `--dr_friction` | ±70% friction (one factor for all geoms) |
| | `--dr_damping` | ±100% joint damping |
| | `--obs_noise` | 0.02 std noise on positions (x10 on velocities) |
| Pushes (during the episode) | `--push_force` | 150 N on the torso, random horizontal direction |
| | `--push_interval`, `--push_duration` | every 3 s on average (exponential), 0.1 s long |

`--log_std_min` (default -1.2) floors the log standard deviation of the policy: without it the policy collapsed to a near-deterministic one, and its PPO updates became erratic (KL spikes). Use `-inf` to disable it.

The CSV and the console also report `Diff` (level), `Succ` (share of finished episodes that survived) and `Eps` (episodes finished in the epoch: when 0, `Ret` and `Len` read 0 because episodes outlast an epoch). The observation contains the absolute heading and position of the root, so the start heading and position are *not* randomized; doing so needs a heading-invariant observation.

Checks of the randomization, curriculum and `log_std` floor: `python3 -m scripts.test_randomization`. To measure robustness: `python3 -m scripts.eval_policy --model <ckpt> --difficulty 0 0.3 0.5` (exact intensity, no random amplitude).

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


