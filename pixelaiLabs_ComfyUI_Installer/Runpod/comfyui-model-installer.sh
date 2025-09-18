#!/usr/bin/env bash
# comfyui‑model‑installer.sh
# Interactive downloader for Flux Dev & Wan models (Linux)

set -euo pipefail

# ──────────────────────────────────────────────
#  Run from the directory that contains this script
# ──────────────────────────────────────────────
cd "$(dirname "$0")"

# ──────────────────────────────────────────────
#  Locate a Python interpreter
# ──────────────────────────────────────────────
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "❌  Python is not installed or not on PATH."
    exit 1
fi

# ──────────────────────────────────────────────
#  Ensure pip is available
# ──────────────────────────────────────────────
if ! command -v pip >/dev/null 2>&1; then
    echo "pip not found – installing with ensurepip…"
    "$PYTHON" -m ensurepip --upgrade
fi

# ──────────────────────────────────────────────
#  Install / upgrade required Python packages
# ──────────────────────────────────────────────
echo "Installing required Python packages (hf_transfer, huggingface_hub)…"
"$PYTHON" -m pip install --quiet --upgrade \
    hf_transfer \
    'huggingface_hub[cli,tqdm]' \
    'huggingface_hub[hf_xet]'

# ──────────────────────────────────────────────
#  Helper: pause until user hits <Enter>
# ──────────────────────────────────────────────
pause() { read -rp "Press Enter to continue…"; }

# ──────────────────────────────────────────────
#  Main menu loop
# ──────────────────────────────────────────────
while true; do
    clear
    cat <<EOF
================================================
        AI MODEL DOWNLOADER LAUNCHER
================================================

Choose a model to download:

1) Flux Dev FP8         (24 GB or more VRAM)
2) Flux Dev GGUF        (24 GB or lower VRAM)
3) Flux Kontext GGUF    (24 GB or lower VRAM)
4) Wan 2.1 GGUF InfiniteTalk     (32 GB VRAM or Less)
5) Wan 2.1 Vace GGUF    (32 GB VRAM or Less)
6) Wan 2.1 Phantom GGUF (32 GB VRAM or Less)
7) NSFW Lessons Models  (Adult Content)
8) Wan 2.2 T2V Models   (Text-to-Video)
9) Wan 2.2 I2V Models   (Image-to-Video)
9) Exit

================================================
EOF
    echo
    read -rp "Enter your choice (1-10): " choice
    echo

    case "$choice" in
        1) 
            echo "🚀 Running Flux Dev FP8 downloader…"
            if [[ -f "Download_fluxDev_models_FP8.py" ]]; then
                "$PYTHON" Download_fluxDev_models_FP8.py
            else
                echo "❌ Error: Download_fluxDev_models_FP8.py not found in current directory."
            fi
            pause ;;
        2) 
            echo "🚀 Running Flux Dev GGUF downloader…"
            if [[ -f "Download_fluxDev_models_GGUF.py" ]]; then
                "$PYTHON" Download_fluxDev_models_GGUF.py
            else
                echo "❌ Error: Download_fluxDev_models_GGUF.py not found in current directory."
            fi
            pause ;;
        3) 
            echo "🚀 Running Flux Kontext GGUF downloader…"
            if [[ -f "Download_models_Flux_Kontext_GGUF.py" ]]; then
                "$PYTHON" Download_models_Flux_Kontext_GGUF.py
            else
                echo "❌ Error: Download_models_Flux_Kontext_GGUF.py not found in current directory."
            fi
            pause ;;
        4) 
            echo "🚀 Running Wan 2.1 GGUF downloader…"
            if [[ -f "Download_models_GGUF.py" ]]; then
                "$PYTHON" Download_models_GGUF.py
            else
                echo "❌ Error: Download_models_GGUF.py not found in current directory."
            fi
            pause ;;
        5) 
            echo "🚀 Running Wan 2.1 Vace GGUF downloader…"
            if [[ -f "Download_models_GGUF_VACE.py" ]]; then
                "$PYTHON" Download_models_GGUF_VACE.py
            else
                echo "❌ Error: Download_models_GGUF_VACE.py not found in current directory."
            fi
            pause ;;
        6) 
            echo "🚀 Running Wan 2.1 Phantom GGUF downloader…"
            if [[ -f "Download_models_GGUF_PHANTOM.py" ]]; then
                "$PYTHON" Download_models_GGUF_PHANTOM.py
            else
                echo "❌ Error: Download_models_GGUF_PHANTOM.py not found in current directory."
            fi
            pause ;;
        7) 
            echo "🚀 Running NSFW Lessons Models downloader…"
            if [[ -f "Download_models_NSFW.py" ]]; then
                "$PYTHON" Download_models_NSFW.py
            else
                echo "❌ Error: Download_models_NSFW.py not found in current directory."
            fi
            pause ;;
        8) 
            echo "🚀 Running Wan 2.2 T2V Models downloader…"
            if [[ -f "Download_wan2-2_T2V.py" ]]; then
                "$PYTHON" Download_wan2-2_T2V.py
            else
                echo "❌ Error: Download_wan2-2_T2V.py not found in current directory."
            fi
            pause ;;
        9) 
            echo "🚀 Running Wan 2.2 I2V Models downloader…"
            if [[ -f "Download_wan2-2_I2V.py" ]]; then
                "$PYTHON" Download_wan2-2_I2V.py
            else
                echo "❌ Error: Download_wan2-2_I2V.py not found in current directory."
            fi
            pause ;;
        10) 
            echo "✅ Exiting..."
            echo "Thank you for using the AI Model Downloader!"
            exit 0 ;;
        *) 
            echo "❌ Invalid choice '$choice'. Please enter a number between 1-9."
            pause ;;
    esac
done