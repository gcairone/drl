# CODEX Technical Report — `gcairone/drl`

## 1) Overall Project Summary

This repository implements a custom Deep Reinforcement Learning (DRL) pipeline for a MuJoCo humanoid using PyTorch, centered on PPO (Proximal Policy Optimization).

High-level workflow:
1. `scripts/train.py` trains PPO in `envs/humanoid_env.py` and logs metrics/checkpoints.
2. `scripts/rollout.py` generates trajectory `.npz` files from a trained policy (or random policy).
3. `scripts/eval.py` computes standing-time statistics from trajectory data.
4. `scripts/plot_log.py` analyzes and plots training CSV metrics.
5. `scripts/play.py` and `scripts/test_mujoco.py` are GUI/viewer-based scripts.

---

## 2) Repository Structure Overview

Top-level directories/files:

- `agents/`
  - `base_agent.py`: abstract RL agent interface (select/update/save/load/policy)
  - `policy.py`: `BasePolicy`, `NullPolicy`, `RandomPolicy`, `TorchPolicy`
  - `ppo.py`: PPO agent with actor/critic networks and update loop
- `envs/`
  - `humanoid_env.py`: MuJoCo humanoid environment wrapper (reset/step/reward/done/render)
- `models/`
  - `base_dnn.py`: reusable MLP (`BaseDNN`) and save/load helpers
  - `actor_critic.py`: `GaussianActor` + `Critic`
- `utils/`
  - `buffer.py`: PPO trajectory buffer + GAE-Lambda + tensor conversion
  - `logger.py`: CSV logger
- `scripts/`
  - `train.py`, `rollout.py`, `eval.py`, `plot_log.py`, `play.py`, `test_dnn.py`, `test_mujoco.py`
- `assets/`
  - `humanoid.xml`: MuJoCo humanoid model definition
- Root docs/notes:
  - `README.md`, `readme.txt`, `todo.txt`, `.gitignore`

No dependency manifest was found (`requirements.txt`, `pyproject.toml`, `setup.py`, `package.json`, etc. are absent).

---

## 3) Main Components / Modules

### Environment (`envs/humanoid_env.py`)
- Loads `assets/humanoid.xml` via MuJoCo API.
- Observation = concatenation of `qpos` + `qvel`.
- Action clipped to `[-1, 1]` and applied as `data.ctrl`.
- Reward: `alive_reward - ctrl_cost` where alive reward is torso height if `z > 1.2`, otherwise 0.
- Termination: torso height `< 0.2`.
- Includes passive viewer replay utilities (`render_init_passive`, `render_sync`, `render_close`) and `_set_state`.

### PPO Agent (`agents/ppo.py`)
- Actor: Gaussian policy (`models.actor_critic.GaussianActor`).
- Critic: scalar value network (`models.actor_critic.Critic`).
- `select_action` returns sampled action + `log_prob` + value.
- `update` performs clipped PPO policy optimization and critic MSE optimization.
- Saves and loads actor/critic + optimizers, can restore training state.

### Models (`models/`)
- `BaseDNN`: configurable MLP with hidden layers and activation.
- `GaussianActor`: Normal distribution with learned mean and global learnable log-std.
- `Critic`: value estimator `V(s)`.

### Data / Logging (`utils/`)
- `PPOBuffer`: stores on-policy transitions, computes GAE + returns, normalizes advantages.
- `Logger`: writes metric rows to CSV.

### Script-level Execution Flow
- `train.py`: create env → collect rollouts each epoch → update PPO → log metrics → save checkpoints.
- `rollout.py`: load checkpoint/random policy → run env steps → save `qpos/qvel/actions/rewards` to `.npz`.
- `eval.py`: read trajectory → compute standing-duration episodes from `qpos[:,2]`.
- `plot_log.py`: read CSV with pandas → print stats → produce matplotlib plots.
- `play.py`: replay saved trajectory in GUI viewer.

---

## 4) Dependency Analysis and Installation Process

### Dependency signals in repository
Inferred from imports:
- Required runtime: `numpy`, `torch`, `mujoco`
- Optional analysis/plotting: `pandas`, `matplotlib`
- Stdlib: `argparse`, `os`, `time`, `datetime`, `csv`, `typing`, etc.

### Installation command executed
```bash
python3 -m pip install --user numpy torch mujoco pandas matplotlib
```

### Installation result
- **Status:** Success
- Installed versions (verified):
  - `numpy 2.4.4`
  - `torch 2.11.0+cu130`
  - `mujoco 3.6.0`
  - `pandas 3.0.2`
  - `matplotlib 3.10.8`
- `torch.cuda.is_available()` returned `False` in this environment.

### Observed installation characteristics
- Installing `torch` pulled a **large dependency tree**, including CUDA-related packages (`cuda-toolkit`, `nvidia-*`, `triton`, etc.), despite no GPU availability.

---

## 5) Execution Results

All commands executed from repository root unless noted.

### 5.1 `scripts/train.py` (headless)
Command:
```bash
python3 -m scripts.train --steps_per_epoch 64 --epochs 1 --max_ep_len 64 --pi_iters 2 --v_iters 2 --save_freq 1 --save_dir /tmp/drl-results/checkpoints --log_path /tmp/drl-results/logs
```
Result:
- **Status:** Success
- Produced one epoch of training, printed metrics, and saved checkpoint:
  - `/tmp/drl-results/checkpoints/2026-04-12_21-46-33/model_epoch_1.pth`

### 5.2 `scripts/rollout.py`
First attempt:
```bash
python3 -m scripts.rollout --model /tmp/drl-results/checkpoints/2026-04-12_21-46-33/model_epoch_1.pth --steps 200 --output /tmp/drl-results/rollouts/humanoid_rollout.npz
```
- **Status:** Failed
- Error: output parent directory missing.

Second attempt (after creating `/tmp/drl-results/rollouts`):
- **Status:** Success
- Output saved: `/tmp/drl-results/rollouts/humanoid_rollout.npz`

### 5.3 `scripts/eval.py`
Command:
```bash
python3 -m scripts.eval --input /tmp/drl-results/rollouts/humanoid_rollout.npz
```
Result:
- **Status:** Success
- Reported:
  - Average standing time: `0.50 s`
  - Stddev: `0.25 s`
  - Episodes found: `2`

### 5.4 `scripts/plot_log.py`
First attempt:
```bash
python3 -m scripts.plot_log --file /tmp/drl-results/logs/2026-04-12_21-46-33_metrics.csv --save /tmp/drl-results/plots/metrics.png
```
- **Status:** Failed
- Error: output parent directory missing.

Second attempt (after creating `/tmp/drl-results/plots`):
- **Status:** Success
- Plot saved to `/tmp/drl-results/plots/metrics.png`

### 5.5 `scripts/play.py` (GUI)
Command:
```bash
python3 -m scripts.play --file /tmp/drl-results/rollouts/humanoid_rollout.npz --fps 60
```
Result:
- **Status:** Failed (expected in headless environment)
- GLFW/X11 error due missing `DISPLAY`.

### 5.6 `scripts/test_mujoco.py` (GUI)
Command:
```bash
python3 -m scripts.test_mujoco
```
Result:
- **Status:** Failed (expected in headless environment)
- GLFW/X11 error due missing `DISPLAY`.

### 5.7 `scripts/test_dnn.py`
Command:
```bash
python3 -m scripts.test_dnn
```
Result:
- **Status:** Failed
- Error: expected directory `results/checkpoints` does not exist before save.

---

## 6) Build Results

- No dedicated build system or build config (Makefile/CI build recipe/packaging build config) was found.
- Therefore, build is **not applicable** as a separate phase.

---

## 7) Test Results

Repository provides script-style tests/demos rather than a formal test suite.

- `scripts/test_dnn.py`: **Failed** (missing output directory)
- `scripts/test_mujoco.py`: **Failed** in headless environment (GUI dependency)
- Functional smoke execution of core pipeline (`train` → `rollout` → `eval` → `plot_log`) succeeded with minimal parameters when output directories existed.

---

## 8) Encountered Errors (Detailed)

1. `scripts/test_dnn.py`
   - Exception: `RuntimeError: Parent directory results/checkpoints does not exist.`
   - Source behavior: script saves model without creating parent directory.

2. `scripts/rollout.py`
   - Exception: `FileNotFoundError: [Errno 2] No such file or directory: '/tmp/drl-results/rollouts/humanoid_rollout.npz'`
   - Cause: script does not create output directory before `np.savez`.

3. `scripts/plot_log.py`
   - Exception: `FileNotFoundError: [Errno 2] No such file or directory: '/tmp/drl-results/plots/metrics.png'`
   - Cause: script does not create output directory before `plt.savefig`.

4. `scripts/play.py` and `scripts/test_mujoco.py`
   - GLFW error: `X11: The DISPLAY environment variable is missing`
   - Followed by: `ERROR: could not initialize GLFW`
   - Cause: GUI viewer requires display server; current environment is headless.

5. `scripts/test_mujoco.py` with `MUJOCO_GL=egl`
   - Exception: `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`
   - Indicates EGL/OpenGL platform initialization issue in this runtime context.

---

## 9) Potential Issues, Risks, and Edge Cases

1. **No pinned dependencies**
   - Missing requirements/lock files creates reproducibility risk across environments.

2. **Output directory assumptions**
   - Multiple scripts assume parent directories exist (`rollout.py`, `plot_log.py`, `test_dnn.py`).

3. **Headless vs GUI mismatch**
   - GUI scripts (`play.py`, `test_mujoco.py`) require display and fail in common CI/headless contexts.

4. **Import path coupling**
   - Scripts rely on repository-root execution/module path assumptions (e.g., `from envs...`).

5. **Checkpoint/log path inconsistency in docs**
   - `README.md` and code defaults differ in places (`checkpoints/` vs `results/checkpoints`, etc.).

6. **Potential runtime path/API issue in `test_mujoco.py`**
   - Script references `env.close()` while environment defines `render_close()`; this may fail if GUI startup succeeds.

7. **Large dependency footprint**
   - Installing latest `torch` can pull large CUDA-related packages even on CPU-only hosts, increasing install time/storage.

---

## 10) Technical Observations and Relevant Insights

- Architecture is modular and easy to follow: env, agent, models, utils, scripts are clearly separated.
- PPO implementation is straightforward and suitable for educational/prototyping use.
- `train.py` supports checkpoint resume and serializes buffer state in checkpoints.
- Reward is minimal (alive torso-height threshold and control penalty), which may limit locomotion quality without further shaping.
- `assets/humanoid.xml` is a complete MuJoCo humanoid model (Apache-2.0 header from DeepMind source lineage).
- Current repository quality is oriented more toward experimentation than production packaging/automation.

---

## 11) Files Reviewed

All repository files and directories were inspected, including:
- Root files: `.gitignore`, `README.md`, `readme.txt`, `todo.txt`
- Python packages/modules under `agents/`, `envs/`, `models/`, `scripts/`, `utils/`
- MuJoCo asset: `assets/humanoid.xml`
- Directory structure under repository root

(Internal `.git/` metadata directories were enumerated for structure context but are not part of application logic.)
