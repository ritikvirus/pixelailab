# --- START OF Download_Wan2.1_models(GGUF).py ---

import os
import sys
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError
)

# --- Configuration ---
def _resolve_models_dir():
    env_dir = os.environ.get("COMFY_MODELS_DIR")
    if env_dir:
        return env_dir
    for base in ("/Workspace/ComfyUI/models", "/workspace/ComfyUI/models"):
        try:
            os.makedirs(base, exist_ok=True)
            return base
        except Exception:
            continue
    fallback = os.path.join(os.getcwd(), "ComfyUI", "models")
    os.makedirs(fallback, exist_ok=True)
    return fallback

BASE_DOWNLOAD_DIR = _resolve_models_dir()
USE_SYMLINKS = False  # Copy files for compatibility

# VRAM-based GGUF model options for unet
VRAM_OPTIONS = {
    "12gb": [
        {"filename": "wan2.1-i2v-14b-480p-Q3_K_M.gguf", "quant": "Q3_K_M"},
        {"filename": "wan2.1-i2v-14b-480p-Q3_K_S.gguf", "quant": "Q3_K_S"},
        {"filename": "wan2.1-i2v-14b-480p-Q4_0.gguf", "quant": "Q4_0"},
        {"filename": "wan2.1-i2v-14b-480p-Q4_1.gguf", "quant": "Q4_1"},
        {"filename": "wan2.1-i2v-14b-480p-Q4_K_M.gguf", "quant": "Q4_K_M"},
        {"filename": "wan2.1-i2v-14b-480p-Q4_K_S.gguf", "quant": "Q4_K_S"}
    ],
    "16gb": [
        {"filename": "wan2.1-i2v-14b-480p-Q5_0.gguf", "quant": "Q5_0"},
        {"filename": "wan2.1-i2v-14b-480p-Q5_1.gguf", "quant": "Q5_1"},
        {"filename": "wan2.1-i2v-14b-480p-Q5_K_M.gguf", "quant": "Q5_K_M"},
        {"filename": "wan2.1-i2v-14b-480p-Q5_K_S.gguf", "quant": "Q5_K_S"}
    ],
    "24gb": [
        {"filename": "wan2.1-i2v-14b-480p-Q6_K.gguf", "quant": "Q6_K"},
        {"filename": "wan2.1-i2v-14b-480p-Q8_0.gguf", "quant": "Q8_0"}
    ]
}

# UMT5 encoder GGUF models for clip, mapped by quantization
UMT5_ENCODER_MODELS = {
    "Q3_K_M": "umt5-xxl-encoder-Q3_K_M.gguf",
    "Q3_K_S": "umt5-xxl-encoder-Q3_K_S.gguf",
    "Q4_0": "umt5-xxl-encoder-Q4_K_M.gguf",  # Closest match for Q4_0
    "Q4_1": "umt5-xxl-encoder-Q4_K_M.gguf",
    "Q4_K_M": "umt5-xxl-encoder-Q4_K_M.gguf",
    "Q4_K_S": "umt5-xxl-encoder-Q4_K_S.gguf",
    "Q5_0": "umt5-xxl-encoder-Q5_K_M.gguf",
    "Q5_1": "umt5-xxl-encoder-Q5_K_M.gguf",
    "Q5_K_M": "umt5-xxl-encoder-Q5_K_M.gguf",
    "Q5_K_S": "umt5-xxl-encoder-Q5_K_S.gguf",
    "Q6_K": "umt5-xxl-encoder-Q6_K.gguf",
    "Q8_0": "umt5-xxl-encoder-Q8_0.gguf"
}

# Base download tasks (for vae and clip_vision)
DOWNLOAD_TASKS = [
        # --- InfiniteTalk ---
    {
        "repo_id": "Kijai/WanVideo_comfy_GGUF",
        "repo_type": "model",
        "filename": "InfiniteTalk/Wan2_1-InfiniteTalk_Multi_Q4_K_M.gguf",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
    },
    {
        "repo_id": "Kijai/WanVideo_comfy_GGUF",
        "repo_type": "model",
        "filename": "InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q4_K_M.gguf",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
    },
    # --- vae ---
    {
        "repo_id": "simwalo/Wan2.1_SkyreelsV2",
        "repo_type": "dataset",
        "filename": "wan_2.1_vae.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "vae")
    },
    {
        "repo_id": "Kijai/WanVideo_comfy",
        "repo_type": "model",
        "filename": "Wan2_1_VAE_bf16.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "vae")
    },
    # --- clip_vision ---
    {
        "repo_id": "simwalo/Wan2.1_SkyreelsV2",
        "repo_type": "dataset",
        "filename": "clip_vision_h.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip_vision")
    },
       # --- clip ---
    {
        "repo_id": "Kijai/WanVideo_comfy",
        "repo_type": "model",
        "filename": "umt5-xxl-enc-bf16.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip")
    },
        # --- diffusion_models---
    {
        "repo_id": "simwalo/Wan2.1_SkyreelsV2",
        "repo_type": "dataset",
        "filename": "wan2.1_t2v_1.3B_fp16.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "diffusion_models")
    },
            # --- LoRA---
    {
        "repo_id": "simwalo/Wan2.1_SkyreelsV2",
        "repo_type": "dataset",
        "filename": "wan_phut_hon_dance.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
        {
        "repo_id": "Kijai/WanVideo_comfy",
        "repo_type": "model",
        "filename": "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
         # --- Controlnet---
            {
        "repo_id": "Kijai/WanVideo_comfy",
        "repo_type": "model",
        "filename": "Wan21_Uni3C_controlnet_fp16.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "controlnet")
    }
]

def get_user_vram_choice():
    """Prompts user to select VRAM and GGUF model, returns selected model and quantization."""
    print("\nSelect your NVIDIA GPU VRAM:")
    print("1. 12GB")
    print("2. 16GB")
    print("3. 24GB")
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice == "1":
            vram = "12gb"
            break
        elif choice == "2":
            vram = "16gb"
            break
        elif choice == "3":
            vram = "24gb"
            break
        print("Invalid choice. Please enter 1, 2, or 3.")

    print(f"\nAvailable GGUF models for {vram.upper()} VRAM:")
    for i, model in enumerate(VRAM_OPTIONS[vram], 1):
        print(f"{i}. {model['filename']} (Quant: {model['quant']})")
    
    while True:
        model_choice = input(f"Enter model number (1-{len(VRAM_OPTIONS[vram])}): ").strip()
        try:
            model_idx = int(model_choice) - 1
            if 0 <= model_idx < len(VRAM_OPTIONS[vram]):
                selected_model = VRAM_OPTIONS[vram][model_idx]
                return selected_model["filename"], selected_model["quant"]
            print(f"Invalid choice. Please enter a number between 1 and {len(VRAM_OPTIONS[vram])}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def download_and_process_item(repo_id, local_dir, filename, use_symlinks=False, repo_type=None):
    """
    Downloads a file from a Hugging Face repository.

    Args:
        repo_id (str): Hugging Face repository ID.
        local_dir (str): Target directory for the file.
        filename (str): Specific file to download.
        use_symlinks (bool): Controls symlink behavior.
        repo_type (str): Type of repository ('dataset', 'model', etc.).

    Returns:
        bool: True if successful, False otherwise.
    """
    print("-" * 20)
    os.makedirs(local_dir, exist_ok=True)

    try:
        print(f"Downloading file:\n  Repo: {repo_id}\n  File: {filename}\n  To:   {local_dir}\n  Symlinks: {use_symlinks}\n  Repo Type: {repo_type or 'default'}")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=use_symlinks,
            resume_download=True,
            repo_type=repo_type
            # token="your_hf_token_here"  # Uncomment and add token if repository is private/gated
        )
        print(f"Successfully downloaded: {file_path}")

        print("-" * 20)
        return True

    except RepositoryNotFoundError as e:
        print(f"\nError: Repository not found for '{repo_id}'. Details: {e}", file=sys.stderr)
    except EntryNotFoundError:
        print(f"\nError: File '{filename}' not found in repo '{repo_id}'.", file=sys.stderr)
    except LocalEntryNotFoundError as e:
        print(f"\nError: Local file system issue for '{repo_id}' in '{local_dir}'. Details: {e}", file=sys.stderr)
    except HfHubHTTPError as e:
        print(f"\nError: HTTP error for repo '{repo_id}'. Status: {e.response.status_code}. Details: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nError: Unexpected error downloading '{filename}' from '{repo_id}': {type(e).__name__} - {e}", file=sys.stderr)

    print("-" * 20)
    return False

def main():
    """Downloads Wan2.1 GGUF models based on user VRAM selection and other specified files."""
    print("Starting Wan2.1 GGUF model downloads...")
    print(f"Base download directory: {os.path.abspath(BASE_DOWNLOAD_DIR)}")
    print(f"Symlinks: {'Enabled' if USE_SYMLINKS else 'Disabled (copying files)'}")

    # Get user's VRAM and model choice
    unet_filename, quant_level = get_user_vram_choice()
    umt5_filename = UMT5_ENCODER_MODELS.get(quant_level, "umt5-xxl-encoder-Q4_K_M.gguf")  # Fallback if quant not found

    # Add unet and umt5 encoder tasks
    tasks = [
        {
            "repo_id": "city96/Wan2.1-I2V-14B-480P-gguf",
            "repo_type": "model",
            "filename": unet_filename,
            "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
        },
        {
            "repo_id": "city96/umt5-xxl-encoder-gguf",
            "repo_type": "model",
            "filename": umt5_filename,
            "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip")
        }
    ] + DOWNLOAD_TASKS  # Combine with base tasks

    successful_downloads = 0
    failed_downloads = 0

    for i, task in enumerate(tasks, 1):
        print(f"\n--- Task {i}/{len(tasks)} ---")
        repo_id = task.get("repo_id")
        local_dir = task.get("local_dir")
        filename = task.get("filename")
        repo_type = task.get("repo_type")

        if not all([repo_id, local_dir, filename]):
            print(f"Error: Task {i} is missing required fields. Skipping.", file=sys.stderr)
            failed_downloads += 1
            continue

        success = download_and_process_item(
            repo_id=repo_id,
            local_dir=local_dir,
            filename=filename,
            use_symlinks=USE_SYMLINKS,
            repo_type=repo_type
        )
        if success:
            successful_downloads += 1
        else:
            failed_downloads += 1
            print(f"Continuing with next task despite error...")

    print("\n" + "=" * 30)
    print("--- Download Summary ---")
    print(f"Selected unet model: {unet_filename} (Quant: {quant_level})")
    print(f"Selected umt5 encoder model: {umt5_filename}")
    print(f"Successful tasks: {successful_downloads}")
    print(f"Failed tasks: {failed_downloads}")
    print("=" * 30)

    if failed_downloads > 0:
        print("\nWarning: One or more tasks failed. Check error messages above.")
        sys.exit(1)
    else:
        print("\nAll tasks completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()

# --- END OF Download_WanSkyReels_models(GGUF).py ---