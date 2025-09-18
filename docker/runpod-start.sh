#!/bin/bash
set -euo pipefail

echo "[runpod-start] Container boot at $(date -u +%FT%TZ)"

# Prefer /Workspace if present; fallback to /workspace. Ensure both paths exist/symlinked.
WORKDIR="/workspace"
if [ -d "/Workspace" ] || mount | grep -q "on /Workspace "; then
  WORKDIR="/Workspace"
fi
mkdir -p "$WORKDIR"
if [ "$WORKDIR" = "/workspace" ]; then
  [ -e "/Workspace" ] || ln -s /workspace /Workspace || true
else
  [ -e "/workspace" ] || ln -s /Workspace /workspace || true
fi

SEED_SRC="/opt/pixelailabs_installer"
SEED_MARK="$WORKDIR/.seeded_from_image"

# Seed once (non-destructive): copy only missing files
if [ ! -f "$SEED_MARK" ]; then
  echo "[runpod-start] Seeding $WORKDIR from image contents..."
  rsync -a "$SEED_SRC/" "$WORKDIR/"
  echo "seeded=$(date -u +%FT%TZ)" > "$SEED_MARK"
fi

# Locate installer script in multiple common locations or by search
INSTALLER_CANDIDATES=(
  "$WORKDIR/pixelaiLabs_ComfyUI_Installer/Runpod/ComfyUI_Installer_Runpod.sh"
  "$WORKDIR/ComfyUI_Installer_Runpod.sh"
  "$WORKDIR/Runpod/ComfyUI_Installer_Runpod.sh"
)
INSTALLER_PATH=""
for p in "${INSTALLER_CANDIDATES[@]}"; do
  if [ -f "$p" ]; then INSTALLER_PATH="$p"; break; fi
done
if [ -z "$INSTALLER_PATH" ]; then
  # Last-resort search up to depth 4
  set +e
  INSTALLER_PATH=$(find "$WORKDIR" -maxdepth 4 -type f -name "ComfyUI_Installer_Runpod.sh" | head -n1)
  set -e
fi

# Install JupyterLab if available (global pip)
ensure_jupyter() {
  if ! command -v jupyter >/dev/null 2>&1; then
    echo "[runpod-start] Installing JupyterLab..."
    python3 -m pip install --no-cache-dir --upgrade pip >/dev/null 2>&1 || true
    python3 -m pip install --no-cache-dir jupyterlab >/dev/null 2>&1 || true
  fi
}

start_jupyter() {
  if command -v jupyter >/dev/null 2>&1; then
    echo "[runpod-start] Starting JupyterLab (port 8888, root=$WORKDIR)"
    nohup jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.notebook_dir="$WORKDIR" >/var/log/jupyterlab.log 2>&1 &
  fi
}

# Ensure JupyterLab available and start FIRST
ensure_jupyter
start_jupyter

# Run installer on first boot (after JupyterLab is up)
BOOT_MARK="$WORKDIR/.installed_comfyui"
if [ ! -f "$BOOT_MARK" ]; then
  if [ -n "$INSTALLER_PATH" ]; then
    echo "[runpod-start] First boot: running installer at $INSTALLER_PATH"
    bash "$INSTALLER_PATH" || echo "[runpod-start] WARNING: Installer returned non-zero."
    echo "installed=$(date -u +%FT%TZ)" > "$BOOT_MARK"
  else
    echo "[runpod-start] WARNING: Installer script not found; proceeding without install."
  fi
else
  echo "[runpod-start] Installer previously completed. Skipping."
fi

# Provide helper alias
ln -sf "$WORKDIR/Run_Comfyui.sh" /usr/local/bin/run-comfyui 2>/dev/null || true

# Start ComfyUI if not already listening on 8188
if ! (ss -ltnp 2>/dev/null | grep -q ":8188"); then
  echo "[runpod-start] Starting ComfyUI (port 8188)"
  if [ -x "$WORKDIR/Run_Comfyui.sh" ]; then
    exec bash "$WORKDIR/Run_Comfyui.sh"
  elif [ -d "$WORKDIR/ComfyUI" ]; then
    cd "$WORKDIR/ComfyUI"
    if [ -f "venv/bin/activate" ]; then source venv/bin/activate || true; fi
    exec python main.py --fast --listen --disable-cuda-malloc
  else
    echo "[runpod-start] ComfyUI not found. Dropping to shell."
    exec bash
  fi
else
  echo "[runpod-start] ComfyUI already running on 8188. Dropping to shell for logs."
  exec bash
fi


