# ── Base image: CUDA 12.4 + cuDNN + Ubuntu 22.04 ─────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Кэш моделей HuggingFace пробросим в volume
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        git \
        wget \
        curl \
        vim \
        ca-certificates \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем pip через встроенный ensurepip (не требует distutils и сети)
RUN python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# ── 1. PyTorch с поддержкой CUDA 12.4 (RTX 3060 — compute 8.6) ───────────────
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ── 2. Остальные зависимости из requirements.txt (torch уже установлен) ───────
COPY requirements.txt .
# Фильтруем строки torch и пишем во временный файл (stdin-pipe ненадёжен в sh)
RUN grep -vE "^torch" requirements.txt > /tmp/req_filtered.txt && \
    pip install -r /tmp/req_filtered.txt && \
    rm /tmp/req_filtered.txt

# ── 3. JupyterLab (заменяет классический jupyter из requirements.txt) ─────────
RUN pip install "jupyterlab>=4.1.0" jupyter-contrib-nbextensions

# ── 4. Конфигурация JupyterLab ────────────────────────────────────────────────
RUN jupyter lab --generate-config && \
    printf '%s\n' \
        "c.ServerApp.ip = '0.0.0.0'" \
        "c.ServerApp.port = 8888" \
        "c.ServerApp.open_browser = False" \
        "c.ServerApp.token = ''" \
        "c.ServerApp.password = ''" \
        "c.ServerApp.allow_root = True" \
        "c.ServerApp.root_dir = '/workspace'" \
        "c.ServerApp.disable_check_xsrf = False" \
    >> /root/.jupyter/jupyter_lab_config.py

EXPOSE 8888

# ── Entrypoint: JupyterLab ────────────────────────────────────────────────────
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
