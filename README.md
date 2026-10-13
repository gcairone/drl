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

### Running task

`--task run` trains the humanoid to run along a commanded direction at a commanded speed (`--task stand`, the default, is the standing task above). It is a separate task with its own observation, reward and curriculum (`envs/run_env.py`):

* **Command**: per episode a speed `v*` (0 to 3.5 m/s) and a heading offset `phi` from the initial facing direction. The command is part of the observation, so one policy runs at any speed and direction.
* **Observation** (76 values): torso height, gravity direction, centre-of-mass velocity and angular velocity in the torso frame, joint angles and velocities, the previous action, and the command. No absolute position or heading: it does not grow with the distance covered and the policy cannot depend on where it is.
* **Actions**: target joint angles tracked by a PD controller (`--pd_control 0` for raw torques). With raw torques the policy learned to fall forward at the commanded speed instead of balancing.
* **Reward** per step: `w_alive` + `w_vel`*exp(-((v_par - v*)/sigma)^2) + `w_prog`*clip(v_par/v*, 0, 1) + `w_head`*(1 + cos(heading error))/2 + `w_up`*exp(-(tilt/0.5)^2) - `w_lat`*v_perp^2 - `w_ctrl`*|u|^2 - `w_rate`*|a - a_prev|^2, with `v_par`/`v_perp` the body velocity along/across the command. All weights are options of `train` (`--w_vel`, ...). The episode ends if the torso drops below `--fall_height` (0.8 m) or tilts beyond `--fall_tilt` (1 rad).
* **Domain randomization and pushes**: those of the standing task (reset randomization, masses, friction, actuator strength, damping, observation noise, random pushes on the torso).
* **Curriculum**: one level in [0, 1], raised when at least 70% of the recent episodes succeed (they do not fall *and* reach 70% of the commanded speed on average) and lowered below 30%. It drives, one after the other: (1) an **assist**, a stabilizing torque on the torso plus an upward force carrying 90% of the weight, full at level 0 and gone at 0.25 (`--assist_end`); (2) the upper bound of the commanded speed, from 0.5 m/s at level 0.1 to `--run_speed` (3.5 m/s) at 0.7; (3) the heading offset, up to `--run_heading` (0.8 rad) from level 0.4; (4) the randomization and pushes, from level 0.5.

Recommended training (8 environments, larger batches and a 256x256 network are the defaults for this task):

```bash
python3 -m scripts.train --task run --num_envs 8 --steps_per_epoch 8192 --minibatch_size 512 \
    --epochs 540 --save_freq 60 --save_dir results/run/checkpoints --log_path results/run/logs
```

The log adds `S_speed`, `S_v_target`, `S_assist`, `S_lat`, `S_head_err`, `S_tilt` and the reward terms (means over the episodes finished in the epoch). To evaluate a checkpoint under a fixed command and fixed perturbations (no assist):

```bash
python3 -m scripts.eval_run --model <ckpt> --speeds 1 2 3 --headings 0 0.5 --perturbation 0 0.5
```

Checks: `python3 -m scripts.test_run_env`. `promote_best` and the `pretrained/` folder are for the standing task.

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


