#!/usr/bin/env bash
# Setup script for the cloud environment.
# Paste the contents into the environment's "Setup script" field (or call this
# file from there). Idempotent: safe to re-run on every new session.
#
# Installs what the Dockerfile installs, minus the X11/GUI tooling:
# numpy, pandas, matplotlib, mujoco, PyTorch.
set -euo pipefail

PIP="python3 -m pip install --quiet --no-cache-dir --break-system-packages"

# CPU-only torch is preferred (much smaller than the default CUDA wheel), but
# download.pytorch.org is blocked by some network policies (403). Fail fast on
# it and fall back to PyPI, which pulls the CUDA build but works everywhere.
if ! python3 -c "import torch" 2>/dev/null; then
  $PIP --retries 0 --timeout 15 torch --index-url https://download.pytorch.org/whl/cpu \
    || $PIP torch
fi

$PIP numpy pandas matplotlib mujoco

# Offscreen rendering (scripts/make_gif.py) uses OSMesa, MuJoCo's software renderer: no GPU or display needed.
if command -v apt-get >/dev/null 2>&1; then
  (apt-get update -qq && apt-get install -y -qq --no-install-recommends libosmesa6) >/dev/null 2>&1 \
    || echo "warning: could not install libosmesa6, scripts/make_gif.py will not work"
fi

# Training/rollout/eval are headless and need no GL. For offscreen rendering
# set MUJOCO_GL=egl (or osmesa) in the environment variables.

python3 - <<'EOF'
import mujoco, numpy, pandas, matplotlib, torch
print(f"mujoco {mujoco.__version__} | numpy {numpy.__version__} | "
      f"pandas {pandas.__version__} | matplotlib {matplotlib.__version__} | "
      f"torch {torch.__version__}")
EOF
