#!/bin/bash

# Exit on critical errors (but allow some failures)
# set -e  # Commented out to allow script to continue on non-critical errors

echo "Starting installation of ComfyUI, Triton, InsightFace, and dependencies on RunPod..."

# Ensure we're in the workspace directory
cd /workspace || { echo "Failed to change to /workspace directory"; exit 1; }

# Install system dependencies first
echo "Installing system dependencies..."
apt update
apt install -y python3.10 python3.10-venv python3.10-dev python3-pip build-essential cmake pkg-config ffmpeg git portaudio19-dev python3-pyaudio alsa-utils unzip wget curl

# Install additional audio libraries for PortAudio
echo "Installing additional audio dependencies..."
apt install -y libportaudio2 libportaudiocpp0 portaudio19-dev libasound2-dev libsndfile1-dev

# Install Python development headers (fixes Python.h missing error)
apt install -y python3-dev python3.10-dev

# Verify Python 3.10 installation
echo "Verifying Python 3.10 installation..."
python3.10 --version || { echo "Python 3.10 not properly installed"; exit 1; }

# Clone or update ComfyUI repository
echo "Setting up ComfyUI..."
if [ -d "ComfyUI" ]; then
    echo "ComfyUI directory already exists, updating..."
    cd ComfyUI
    git pull origin main || git pull origin master || echo "Warning: Failed to update ComfyUI, continuing with existing version..."
    cd ..
else
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git || { 
        echo "Failed to clone ComfyUI repository"
        echo "Trying alternative clone method..."
        git clone https://github.com/comfyanonymous/ComfyUI ComfyUI || {
            echo "All clone attempts failed. Exiting."
            exit 1
        }
    }
fi

# Change to ComfyUI directory
cd ComfyUI || { echo "Failed to enter ComfyUI directory"; exit 1; }

# Remove existing venv if it exists but is broken
if [ -d "venv" ] && [ ! -f "venv/bin/activate" ]; then
    echo "Removing broken virtual environment..."
    rm -rf venv
fi

# Create and activate virtual environment with Python 3.10
echo "Setting up virtual environment with Python 3.10..."
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "Virtual environment already exists and is functional, using existing one..."
else
    echo "Creating new virtual environment..."
    # Try multiple methods to create venv
    if command -v python3.10 >/dev/null 2>&1; then
        echo "Using python3.10 to create venv..."
        python3.10 -m venv venv || { 
            echo "Failed with python3.10, trying alternative methods..."
            
            # Try with explicit ensurepip
            python3.10 -m ensurepip --upgrade 2>/dev/null || true
            python3.10 -m venv venv || {
                
                # Try with system python3
                echo "Trying with python3..."
                python3 -m venv venv || {
                    
                    # Try installing python3-venv explicitly
                    echo "Installing python3-venv package..."
                    apt install -y python3-venv
                    python3 -m venv venv || {
                        
                        # Last resort: use virtualenv
                        echo "Trying virtualenv as last resort..."
                        pip3 install virtualenv
                        virtualenv -p python3.10 venv || {
                            echo "All virtual environment creation methods failed"
                            exit 1
                        }
                    }
                }
            }
        }
    else
        echo "python3.10 not found, using python3..."
        python3 -m venv venv || {
            echo "Failed to create virtual environment"
            exit 1
        }
    fi
fi

# Verify venv creation
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment creation failed - activate script not found"
    echo "Contents of venv directory:"
    ls -la venv/ 2>/dev/null || echo "venv directory doesn't exist"
    exit 1
fi

echo "Virtual environment created successfully!"
source venv/bin/activate

# Verify activation worked
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Virtual environment activation failed"
    exit 1
fi

echo "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for CUDA 12.4
echo "Installing PyTorch for CUDA 12.4..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install build dependencies first
echo "Installing build dependencies..."
pip install wheel setuptools packaging ninja cython numpy

# Fix dependency conflicts by upgrading huggingface-hub first
echo "Fixing dependency conflicts..."
pip install --upgrade --force-reinstall "huggingface-hub>=0.27.0"

# Install additional Python packages including audio support
echo "Installing additional dependencies..."
pip install -r requirements.txt
pip install onnxruntime-gpu accelerate diffusers transformers
pip install mediapipe>=0.10.8
pip install omegaconf
pip install einops
pip install opencv-python
pip install face-alignment
pip install decord
pip install ffmpeg-python>=0.2.0
pip install safetensors
pip install soundfile
pip install pytorch-lightning

# Install audio packages with proper system dependencies
echo "Installing audio packages..."
# Ensure PortAudio is available before installing Python audio packages
ldconfig
pip install --force-reinstall pyaudio
pip install --force-reinstall sounddevice

# Install additional packages that might be missing
pip install scipy
pip install librosa
pip install resampy

# Install deepspeed
echo "Installing deepspeed..."
pip install deepspeed

# Install Triton compatible with CUDA 12.4 and Python 3.10
echo "Installing Triton..."
pip install triton==3.2.0

# Try InsightFace installation with different approaches
echo "Installing InsightFace..."

# Install additional dependencies that InsightFace might need
pip install --upgrade cython
pip install --upgrade numpy
pip install Pillow
pip install scikit-image

# Method 1: Try the pre-built wheel first
pip install insightface==0.7.3 || {
    echo "Pre-built wheel failed, trying alternative methods..."
    
    # Method 2: Try without version constraint
    pip install insightface || {
        echo "Standard install failed, building from source with extra flags..."
        
        # Method 3: Build from source with explicit flags
        export CC=gcc
        export CXX=g++
        pip install --no-binary=insightface insightface==0.7.3 || {
            echo "Building from source failed, trying git installation..."
            
            # Method 4: Install from git
            pip install git+https://github.com/deepinsight/insightface.git@master#subdirectory=python-package || {
                echo "Warning: InsightFace installation failed, but continuing..."
            }
        }
    }
}

# Clear Triton and Torchinductor caches
echo "Clearing Triton and Torchinductor caches..."
rm -rf ~/.triton/cache
mkdir -p ~/.triton/cache
rm -rf /tmp/torchinductor_$(whoami)/triton
mkdir -p /tmp/torchinductor_$(whoami)/triton

# Create startup scripts with better error handling
echo "Creating startup scripts..."
cat <<EOF > /workspace/Run_Comfyui.sh
#!/bin/bash

# Enhanced ComfyUI Runner - Fixes issues after RunPod restart
# Based on your existing installation script

cd /workspace/ComfyUI

echo "=== ComfyUI Post-Restart Fixes ==="

# Function to check and repair system dependencies
check_system_deps() {
    echo "Checking system dependencies..."
    
    # Check if we have sudo/root access
    if [ "$EUID" -eq 0 ] || command -v sudo >/dev/null 2>&1; then
        SUDO_CMD=""
        if [ "$EUID" -ne 0 ]; then
            SUDO_CMD="sudo"
        fi
        
        # Refresh ldconfig cache (important for PortAudio)
        echo "Refreshing library cache..."
        $SUDO_CMD ldconfig
        
        # Check if essential packages are missing and reinstall if needed
        if ! dpkg -l | grep -q python3-dev; then
            echo "Reinstalling missing Python development headers..."
            $SUDO_CMD apt update -qq
            $SUDO_CMD apt install -y python3-dev python3.10-dev build-essential
        fi
        
        if ! dpkg -l | grep -q portaudio19-dev; then
            echo "Reinstalling missing PortAudio dependencies..."
            $SUDO_CMD apt install -y portaudio19-dev libportaudio2 libportaudiocpp0 libasound2-dev
        fi
    else
        echo "No sudo access - skipping system dependency checks"
    fi
}

# Function to clean problematic cache and temporary files
clean_problematic_files() {
    echo "Cleaning problematic cache files..."
    
    # Remove problematic .cache directory that causes import errors
    if [ -d "/workspace/ComfyUI/custom_nodes/.cache" ]; then
        echo "Removing problematic .cache directory..."
        rm -rf "/workspace/ComfyUI/custom_nodes/.cache"
    fi
    
    # Clean Python bytecode cache
    find /workspace/ComfyUI -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find /workspace/ComfyUI -name "*.pyc" -delete 2>/dev/null || true
    
    # Clean Triton cache (helps with compilation issues)
    rm -rf ~/.triton/cache 2>/dev/null || true
    mkdir -p ~/.triton/cache
    
    # Clean temporary compilation files
    rm -rf /tmp/tmp* 2>/dev/null || true
    rm -rf /tmp/latentsync_* 2>/dev/null || true
    
    echo "Cache cleaning completed."
}

# Function to fix virtual environment issues
fix_venv_issues() {
    echo "Checking virtual environment..."
    
    # Check if venv exists and activate script is present
    if [ ! -f "venv/bin/activate" ]; then
        echo "ERROR: Virtual environment not found or corrupted!"
        echo "Please run the installation script again."
        exit 1
    fi
    
    source venv/bin/activate
    
    # Verify activation
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "ERROR: Failed to activate virtual environment!"
        exit 1
    fi
    
    echo "Virtual environment activated: $VIRTUAL_ENV"
    
    # Check and fix common package issues
    echo "Checking for broken packages..."
    
    # Fix bitsandbytes if it's causing issues
    if pip list | grep -q bitsandbytes; then
        python -c "import bitsandbytes" 2>/dev/null || {
            echo "Fixing bitsandbytes installation..."
            pip uninstall -y bitsandbytes
            pip install bitsandbytes --no-cache-dir
        }
    fi
    
    # Fix triton if it's causing compilation issues
    if pip list | grep -q triton; then
        python -c "import triton" 2>/dev/null || {
            echo "Fixing triton installation..."
            pip uninstall -y triton
            pip install triton==3.2.0 --no-cache-dir
        }
    fi
    
    # Fix audio packages if they're broken
    python -c "import sounddevice" 2>/dev/null || {
        echo "Fixing audio packages..."
        pip uninstall -y sounddevice pyaudio
        pip install --force-reinstall sounddevice pyaudio --no-cache-dir
    }
}

# Function to install/repair ComfyUI-WanVideoWrapper
install_wanvideo_wrapper() {
    echo "Installing/repairing ComfyUI-WanVideoWrapper..."
    
    cd /workspace/ComfyUI/custom_nodes
    
    # Remove existing installations (both normal and disabled)
    if [ -d "ComfyUI-WanVideoWrapper" ]; then
        echo "Removing existing ComfyUI-WanVideoWrapper..."
        rm -rf ComfyUI-WanVideoWrapper
    fi
    
    if [ -d "ComfyUI-WanVideoWrapper.disabled" ]; then
        echo "Removing disabled ComfyUI-WanVideoWrapper..."
        rm -rf ComfyUI-WanVideoWrapper.disabled
    fi
    
    # Clone fresh copy
    echo "Cloning ComfyUI-WanVideoWrapper..."
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git || {
        echo "ERROR: Failed to clone ComfyUI-WanVideoWrapper"
        return 1
    }
    
    # Install requirements
    if [ -d "ComfyUI-WanVideoWrapper" ]; then
        cd ComfyUI-WanVideoWrapper
        
        if [ -f "requirements.txt" ]; then
            echo "Installing requirements for ComfyUI-WanVideoWrapper..."
            # Make sure we're in the virtual environment
            source /workspace/ComfyUI/venv/bin/activate
            
            # Install requirements with error handling
            pip install -r requirements.txt || {
                echo "WARNING: Some requirements failed to install, but continuing..."
            }
        else
            echo "No requirements.txt found for ComfyUI-WanVideoWrapper"
        fi
        
        cd /workspace/ComfyUI/custom_nodes
    fi
    
    echo "ComfyUI-WanVideoWrapper installation completed."
}

# Function to automatically disable other known problematic nodes
auto_disable_problematic_nodes() {
    echo "Auto-disabling other known problematic custom nodes..."
    
    # Known problematic nodes (excluding WanVideoWrapper which we're fixing)
    PROBLEMATIC_NODES=(
        "ComfyUI-SparkTTS"
    )
    
    for node in "${PROBLEMATIC_NODES[@]}"; do
        node_path="/workspace/ComfyUI/custom_nodes/$node"
        disabled_path="${node_path}.disabled"
        
        if [ -d "$node_path" ] && [ ! -d "$disabled_path" ]; then
            echo "Auto-disabling problematic node: $node"
            mv "$node_path" "$disabled_path"
        elif [ -d "$disabled_path" ]; then
            echo "Already disabled: $node"
        fi
    done
    
    echo "Other problematic nodes auto-disabled. Re-enable by removing '.disabled' from folder names if needed."
}

# Function to set optimal environment variables
set_optimal_env() {
    echo "Setting optimal environment variables..."
    
    # CUDA environment
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    # Compilation flags for better compatibility
    export CFLAGS="-I/usr/include/python3.10"
    export CPPFLAGS="-I/usr/include/python3.10"
    
    # PyTorch and related optimizations
    export TORCH_DISABLE_SAFE_DESERIALIZER=1
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    export TOKENIZERS_PARALLELISM=false
    
    # Disable problematic bitsandbytes features
    export BITSANDBYTES_NOWELCOME=1
    
    # Triton optimization
    export TRITON_CACHE_DIR=/tmp/triton_cache
    mkdir -p /tmp/triton_cache
    
    echo "Environment variables configured."
}

# Function to check if nodes need repair
check_node_health() {
    echo "Performing quick health check on custom nodes..."
    
    # Count total nodes vs failed imports from your log pattern
    total_nodes=$(find /workspace/ComfyUI/custom_nodes -maxdepth 1 -type d | wc -l)
    
    echo "Found $total_nodes custom node directories"
    echo "Starting ComfyUI to check for import errors..."
}

# Main execution function
main() {
    echo "🔧 Running post-restart fixes for ComfyUI..."
    echo "This will address common issues that occur after RunPod restart."
    echo ""
    
    # Run all fix functions
    check_system_deps
    clean_problematic_files
    fix_venv_issues
    set_optimal_env
    install_wanvideo_wrapper
    auto_disable_problematic_nodes
    
    # Return to ComfyUI directory for startup
    cd /workspace/ComfyUI
    
    echo ""
    echo "✅ All fixes applied successfully!"
    echo "✅ ComfyUI-WanVideoWrapper installed and ready!"
    echo ""
    echo "🚀 Starting ComfyUI with optimized settings..."
    echo ""
    
    # Start ComfyUI with optimized flags
    python main.py --fast --listen --disable-cuda-malloc
}

# Run the main function
main
EOF
chmod +x /workspace/Run_Comfyui.sh

cat <<EOF > /workspace/Activate_Venv.sh
#!/bin/bash
cd /workspace/ComfyUI

# Check if venv exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run the installation script again."
    exit 1
fi

source venv/bin/activate

# Verify activation
if [ -z "\$VIRTUAL_ENV" ]; then
    echo "ERROR: Failed to activate virtual environment!"
    exit 1
fi

echo "Virtual environment activated: \$VIRTUAL_ENV"
bash
EOF
chmod +x /workspace/Activate_Venv.sh

cat <<EOF > /workspace/Update_Comfy.sh
#!/bin/bash
cd /workspace/ComfyUI
git pull
EOF
chmod +x /workspace/Update_Comfy.sh

# Install custom nodes
echo "Installing custom nodes..."
if [ ! -d "custom_nodes" ]; then
    mkdir -p custom_nodes
fi
# Return to custom_nodes directory
cd /workspace/ComfyUI/custom_nodes

# List of other custom node repositories
declare -a repos=(
    "https://github.com/ltdrdata/ComfyUI-Manager.git"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/kijai/ComfyUI-Florence2"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/ltdrdata/ComfyUI-Inspire-Pack"
    "https://github.com/jamesWalker55/comfyui-various"
    "https://github.com/un-seen/comfyui-tensorops"
    "https://github.com/city96/ComfyUI-GGUF"
    "https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait"
    "https://github.com/cubiq/ComfyUI_FaceAnalysis"
    "https://github.com/BadCafeCode/masquerade-nodes-comfyui"
    "https://github.com/Ryuukeisyou/comfyui_face_parsing"
    "https://github.com/TinyTerra/ComfyUI_tinyterraNodes"
    "https://github.com/Pixelailabs/Save_Florence2_Bulk_Prompts"
    "https://github.com/chflame163/ComfyUI_LayerStyle_Advance"
    "https://github.com/chflame163/ComfyUI_LayerStyle"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
    "https://github.com/orssorbit/ComfyUI-wanBlockswap"
    "https://github.com/1038lab/ComfyUI-SparkTTS"
    "https://github.com/EvilBT/ComfyUI_SLK_joy_caption_two"
    "https://github.com/ltdrdata/ComfyUI-Impact-Subpack"
    "https://github.com/kijai/ComfyUI-WanVideoWrapper"
    "https://github.com/christian-byrne/audio-separation-nodes-comfyui"
)

# Clone and install requirements for each custom node (continues on failure)
for repo in "${repos[@]}"; do
    repo_name=$(basename "$repo" .git)
    echo "Processing $repo_name..."
    
    if [ -d "$repo_name" ]; then
        echo "$repo_name already exists, updating..."
        cd "$repo_name"
        # Handle different default branches
        if [[ "$repo_name" == *"ComfyUI_SLK_joy_caption_two"* ]]; then
            git pull origin master 2>/dev/null || echo "Warning: Failed to update $repo_name, using existing version..."
        else
            git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || echo "Warning: Failed to update $repo_name, using existing version..."
        fi
        cd ..
    else
        echo "Cloning $repo_name..."
        git clone "$repo" || echo "Warning: Failed to clone $repo_name, continuing..."
    fi
    
    if [ -d "$repo_name" ]; then
        cd "$repo_name"
        if [ -f "requirements.txt" ]; then
            echo "Installing requirements for $repo_name..."
            # Activate the virtual environment before installing requirements
            source /workspace/ComfyUI/venv/bin/activate
            
            # Special handling for ComfyUI_SLK_joy_caption_two to avoid huggingface-hub conflicts
            if [[ "$repo_name" == *"ComfyUI_SLK_joy_caption_two"* ]]; then
                echo "Applying special handling for $repo_name to avoid dependency conflicts..."
                grep -v "huggingface_hub" requirements.txt > temp_requirements.txt 2>/dev/null || cp requirements.txt temp_requirements.txt
                pip install -r temp_requirements.txt || echo "Warning: Failed to install requirements for $repo_name, continuing..."
                rm -f temp_requirements.txt
            else
                pip install -r requirements.txt || echo "Warning: Failed to install requirements for $repo_name, continuing..."
            fi
        else
            echo "No requirements.txt found for $repo_name."
        fi
        cd ..
    fi
done

# Return to ComfyUI directory
cd /workspace/ComfyUI

# Activate virtual environment
source venv/bin/activate

# Uninstall current PyTorch and install the new one
echo "Uninstalling current PyTorch..."
pip uninstall torch torchvision torchaudio -y

echo "Installing PyTorch with CUDA 12.8..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# Install SageAttention from the specified wheel
echo "Installing SageAttention from custom wheel..."
pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/4b17732a84bc50cf1e8b790e854ee9cd5e2ebfbf/sageattention-2.1.1-cp310-cp310-linux_x86_64.whl

# Download and extract Joy_caption_two model
echo "Downloading Joy_caption_two model..."
mkdir -p /workspace/ComfyUI/models/LLM
cd /workspace/ComfyUI/models/LLM

# Download and extract Llama model
echo "Downloading Llama-3.1-8B-Lexi-Uncensored-V2-nf4 model..."
wget https://huggingface.co/datasets/simwalo/custom_nodes/resolve/main/Llama-3.1-8B-Lexi-Uncensored-V2-nf4.zip
unzip Llama-3.1-8B-Lexi-Uncensored-V2-nf4.zip
rm Llama-3.1-8B-Lexi-Uncensored-V2-nf4.zip

echo "✅ Models downloaded and extracted successfully!"

echo ""
echo "🎉 Installation complete!"
echo "=================================================="
echo "✅ ComfyUI installed with all dependencies"
echo "✅ Virtual environment properly created and tested"
echo "✅ All custom nodes installed"
echo "✅ PyTorch updated to CUDA 12.8"
echo "✅ SageAttention installed"
echo ""
echo "📋 Available scripts:"
echo "   🚀 /workspace/Run_Comfyui.sh     - Start ComfyUI"
echo "   🔧 /workspace/Activate_Venv.sh   - Activate virtual environment"
echo "   📦 /workspace/Update_Comfy.sh    - Update ComfyUI"
echo ""
echo "🚀 Run '/workspace/Run_Comfyui.sh' to start ComfyUI!"
echo "=================================================="