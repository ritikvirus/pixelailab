FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Base system deps and Python 3.10 for ComfyUI venv (kept outside /workspace)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl wget unzip ca-certificates rsync nano ffmpeg iproute2 \
    build-essential cmake pkg-config \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    libportaudio2 libportaudiocpp0 portaudio19-dev libasound2-dev libsndfile1-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# Bake the whole repo so scripts are available in the image (NOT in /workspace)
COPY . /opt/pixelailabs_installer

RUN echo "built=$(date -u +%FT%TZ)" > /opt/pixelailabs_installer/.image-built.txt \
 && chmod -R a+rX /opt/pixelailabs_installer

# Runtime start script that copies to /workspace and runs installer or starts ComfyUI
COPY docker/runpod-start.sh /usr/local/bin/runpod-start
RUN chmod +x /usr/local/bin/runpod-start

# Pre-install JupyterLab for faster first boot (uses system python3)
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir jupyterlab

EXPOSE 8188 8888
WORKDIR /root
CMD ["/usr/local/bin/runpod-start"]


