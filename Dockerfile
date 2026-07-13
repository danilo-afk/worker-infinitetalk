# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    build-essential \
    g++ \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

WORKDIR /

# Install Python runtime dependencies (hf_transfer p/ download rápido dos pesos)
RUN uv pip install runpod requests websocket-client "huggingface_hub[hf_transfer]" hf_transfer
# comfy-cli não puxa o requirements.txt COMPLETO do ComfyUI → deps novas (sqlalchemy, PIL, comfy_aimdo, etc.)
# faltam e o ComfyUI crasha na subida. Instalar com pip NORMAL (não `uv pip`, não `--no-deps`): mantém o
# torch CUDA já instalado pelo `comfy install --nvidia` (linha `torch` sem versão = satisfeita, não reinstala).
# Abordagem do worker 10eros (LTX-2.3 provado).
RUN if [ -f /comfyui/requirements.txt ]; then /opt/venv/bin/pip install -q --root-user-action=ignore -r /comfyui/requirements.txt; fi
# O requirements.txt (torch sem versão) puxa um torch de CUDA nova demais p/ o driver da GPU do RunPod
# (RTX 4090 driver 12.8 = 12080 → "NVIDIA driver too old" / CUDA init failed). Força torch p/ CUDA 12.6 (compatível).
RUN uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Add application code and scripts
ADD src/start.sh src/network_volume.py handler.py test_input.json ./
# Templates de workflow LTX-2.3 (T2V/I2V) usados pelo modo-prompt do handler
COPY src/workflows /workflows
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

ENV PIP_NO_INPUT=1

# Helper script to switch Manager network mode
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# ============ Custom nodes: forks PRÓPRIOS pinados por commit ============
# Forkados p/ danilo-afk (não depender do HEAD de terceiros — evita quebra por
# breaking change/rename/deleção upstream). Atualizar = bumpar o SHA aqui.

# WanVideoWrapper (fork de kijai) — Wan2.x + InfiniteTalk/MultiTalk (áudio->vídeo),
# Whisper/Wav2Vec embeds, MelBandRoformer. É o node principal do talking-avatar.
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/danilo-afk/ComfyUI-WanVideoWrapper.git && \
    cd ComfyUI-WanVideoWrapper && git checkout 088128b224242e110d3906c6750e9a3a348a659b && \
    uv pip install --no-cache-dir -r requirements.txt

# KJNodes (fork de kijai) — utilitários usados pelos workflows do WanVideoWrapper
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/danilo-afk/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && git checkout e27a505b3ba6ce42687fe00500deda103d9d6071 && \
    uv pip install --no-cache-dir -r requirements.txt

# VideoHelperSuite (fork de Kosinkadink) — output de vídeo (VHS_VideoCombine) + LoadAudio
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/danilo-afk/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && git checkout 4ee72c065db22c9d96c2427954dc69e7b908444b && \
    uv pip install --no-cache-dir -r requirements.txt
# ==========================================

CMD ["/start.sh"]
