# ---------- Dockerfile ----------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1 \
    libglew-dev \
    libosmesa6-dev \
    xvfb \
    x11-utils \
    x11-apps \
    nano \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Aggiorna pip
RUN pip install --upgrade pip --no-cache-dir

# Librerie Python principali
RUN pip install --no-cache-dir numpy pandas matplotlib

# MuJoCo
ENV MUJOCO_VERSION=3.3.7
RUN mkdir -p /root/.mujoco && \
    wget https://github.com/deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz && \
    tar -xzf /tmp/mujoco.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco.tar.gz

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco-${MUJOCO_VERSION}
ENV LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Python MuJoCo
RUN pip install --no-cache-dir mujoco

# PyTorch CPU-only
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ---------- Creazione utente dinamico ----------
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} hostgroup && \
    useradd -m -u ${UID} -g ${GID} hostuser

USER hostuser
WORKDIR /workspace/drl

CMD ["bash"]

