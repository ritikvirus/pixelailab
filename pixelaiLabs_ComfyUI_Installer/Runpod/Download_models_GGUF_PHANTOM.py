# --- START OF Download_models_GGUF_PHANTOM_RunPod.py ---

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

# VRAM-based GGUF model options for unet (Phantom Wan models)
VRAM_OPTIONS = {
    "12gb": [
        {"filename": "Phantom_Wan_14B-Q3_K_S.gguf", "quant": "Q3_K_S"},
        {"filename": "Phantom_Wan_14B-Q3_K_M.gguf", "quant": "Q3_K_M"},
        {"filename": "Phantom_Wan_14B-Q4_0.gguf", "quant": "Q4_0"},
        {"filename": "Phantom_Wan_14B-Q4_1.gguf", "quant": "Q4_1"},
        {"filename": "Phantom_Wan_14B-Q4_K_S.gguf", "quant": "Q4_K_S"},
        {"filename": "Phantom_Wan_14B-Q4_K_M.gguf", "quant": "Q4_K_M"}
    ],
    "16gb": [
        {"filename": "Phantom_Wan_14B-Q5_0.gguf", "quant": "Q5_0"},
        {"filename": "Phantom_Wan_14B-Q5_1.gguf", "quant": "Q5_1"},
        {"filename": "Phantom_Wan_14B-Q5_K_S.gguf", "quant": "Q5_K_S"},
        {"filename": "Phantom_Wan_14B-Q5_K_M.gguf", "quant": "Q5_K_M"}
    ],
    "24gb": [
        {"filename": "Phantom_Wan_14B-Q6_K.gguf", "quant": "Q6_K"},
        {"filename": "Phantom_Wan_14B-Q8_0.gguf", "quant": "Q8_0"}
    ],
    "32gb": [
        {"filename": "Phantom_Wan_14B-F16.gguf", "quant": "F16"},
        {"filename": "Phantom_Wan_14B-BF16.gguf", "quant": "BF16"}
    ]
}

# UMT5 encoder GGUF models for clip, mapped by quantization
UMT5_ENCODER_MODELS = {
    "Q3_K_S": "umt5-xxl-encoder-Q3_K_S.gguf",
    "Q3_K_M": "umt5-xxl-encoder-Q3_K_M.gguf",
    "Q4_0": "umt5-xxl-encoder-Q4_K_M.gguf",  # Closest match for Q4_0
    "Q4_1": "umt5-xxl-encoder-Q4_K_M.gguf",
    "Q4_K_S": "umt5-xxl-encoder-Q4_K_S.gguf",
    "Q4_K_M": "umt5-xxl-encoder-Q4_K_M.gguf",
    "Q5_0": "umt5-xxl-encoder-Q5_K_M.gguf",
    "Q5_1": "umt5-xxl-encoder-Q5_K_M.gguf",
    "Q5_K_S": "umt5-xxl-encoder-Q5_K_S.gguf",
    "Q5_K_M": "umt5-xxl-encoder-Q5_K_M.gguf",
    "Q6_K": "umt5-xxl-encoder-Q6_K.gguf",
    "Q8_0": "umt5-xxl-encoder-Q8_0.gguf",
    "F16": "umt5-xxl-encoder-Q8_0.gguf",  # Use highest quality for F16
    "BF16": "umt5-xxl-encoder-Q8_0.gguf"
}

# Base download tasks (for vae, clip_vision, diffusion_models, and loras)
DOWNLOAD_TASKS = [
    # --- vae ---
    {
        "repo_id": "simwalo/Wan2.1_SkyreelsV2",
        "repo_type": "dataset",
        "filename": "wan_2.1_vae.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "vae")
    },
    # --- clip_vision ---
    {
        "repo_id": "simwalo/Wan2.1_SkyreelsV2",
        "repo_type": "dataset",
        "filename": "clip_vision_h.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip_vision")
    },
        # --- Lora ---
    {
        "repo_id": "Kijai/WanVideo_comfy",
        "repo_type": "model",
        "filename": "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    }
]

def get_user_vram_choice():
    """Prompts user to select VRAM and GGUF model, returns selected model and quantization."""
    print("\nSelect your NVIDIA GPU VRAM:")
    print("1. 12GB")
    print("2. 16GB")
    print("3. 24GB")
    print("4. 32GB+")
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice == "1":
            vram = "12gb"
            break
        elif choice == "2":
            vram = "16gb"
            break
        elif choice == "3":
            vram = "24gb"
            break
        elif choice == "4":
            vram = "32gb"
            break
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

    print(f"\nAvailable Phantom Wan GGUF models for {vram.upper()} VRAM:")
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
    """Downloads Phantom Wan GGUF models based on user VRAM selection and other specified files."""
    print("Starting Phantom Wan GGUF model downloads for RunPod...")
    print("Phantom: Subject-Consistent Video Generation for character identity preservation")
    print(f"Base download directory: {os.path.abspath(BASE_DOWNLOAD_DIR)}")
    print(f"Symlinks: {'Enabled' if USE_SYMLINKS else 'Disabled (copying files)'}")

    # Get user's VRAM and model choice
    unet_filename, quant_level = get_user_vram_choice()
    umt5_filename = UMT5_ENCODER_MODELS.get(quant_level, "umt5-xxl-encoder-Q4_K_M.gguf")  # Fallback if quant not found

    # Add unet and umt5 encoder tasks
    tasks = [
        {
            "repo_id": "QuantStack/Phantom_Wan_14B-GGUF",
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
    print(f"Selected Phantom unet model: {unet_filename} (Quant: {quant_level})")
    print(f"Selected umt5 encoder model: {umt5_filename}")
    print(f"Successful tasks: {successful_downloads}")
    print(f"Failed tasks: {failed_downloads}")
    print("=" * 30)

    if failed_downloads > 0:
        print("\nWarning: One or more tasks failed. Check error messages above.")
        sys.exit(1)
    else:
        print("\nAll tasks completed successfully!")
        print("\nPhantom Wan Model Info:")
        print("- Subject-consistent video generation with character identity preservation")
        print("- Use up to 4 reference images for consistent character appearance")
        print("- Trained on 24fps data, works with 16fps (with slight quality decline)")
        print("- Recommended for horizontal videos for better stability")
        sys.exit(0)

if __name__ == "__main__":
    main()

# --- END OF Download_models_GGUF_PHANTOM_RunPod.py ---