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
BASE_DOWNLOAD_DIR = "ComfyUI/models"
USE_SYMLINKS = False  # Copy files for compatibility

# VRAM-based GGUF model options for unet (flux1-dev and flux1-kontext-dev)
VRAM_OPTIONS = {
    "8gb": [
        {"filename": "flux1-dev-Q3_K_S.gguf", "quant": "Q3_K_S"},
        {"filename": "flux1-kontext-dev-Q3_K_M.gguf", "quant": "Q3_K_M"},
        {"filename": "flux1-kontext-dev-Q4_0.gguf", "quant": "Q4_0"},
        {"filename": "flux1-kontext-dev-Q4_1.gguf", "quant": "Q4_1"}
    ],
    "12gb": [
        {"filename": "flux1-kontext-dev-Q4_K_M.gguf", "quant": "Q4_K_M"},
        {"filename": "flux1-kontext-dev-Q4_K_S.gguf", "quant": "Q4_K_S"},
        {"filename": "flux1-kontext-dev-Q5_0.gguf", "quant": "Q5_0"}
    ],
    "16gb": [
        {"filename": "flux1-kontext-dev-Q5_1.gguf", "quant": "Q5_1"},
        {"filename": "flux1-kontext-dev-Q5_K_M.gguf", "quant": "Q5_K_M"},
        {"filename": "flux1-kontext-dev-Q5_K_S.gguf", "quant": "Q5_K_S"},
        {"filename": "flux1-kontext-dev-Q6_K.gguf", "quant": "Q6_K"}
    ],
    "24gb": [
        {"filename": "flux1-kontext-dev-Q8_0.gguf", "quant": "Q8_0"}
    ]
}

# T5 encoder GGUF models for clip, mapped by quantization
T5_ENCODER_MODELS = {
    "Q3_K_S": "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
    "Q4_0": "t5-v1_1-xxl-encoder-Q4_K_M.gguf",  # Closest match for Q4_0
    "Q4_1": "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
    "Q4_K_S": "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
    "Q5_0": "t5-v1_1-xxl-encoder-Q5_K_S.gguf",
    "Q5_1": "t5-v1_1-xxl-encoder-Q5_K_S.gguf",
    "Q5_K_S": "t5-v1_1-xxl-encoder-Q5_K_S.gguf",
    "Q6_K": "t5-v1_1-xxl-encoder-Q6_K.gguf",
    "Q8_0": "t5-v1_1-xxl-encoder-Q8_0.gguf"
}

def get_user_vram_choice():
    """Prompts user to select VRAM and GGUF model, returns selected model filename and quantization."""
    print("\nSelect your NVIDIA GPU VRAM:")
    print("1. 8GB")
    print("2. 12GB")
    print("3. 16GB")
    print("4. 24GB")
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice == "1":
            vram = "8gb"
            break
        elif choice == "2":
            vram = "12gb"
            break
        elif choice == "3":
            vram = "16gb"
            break
        elif choice == "4":
            vram = "24gb"
            break
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

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

def download_file(repo_id, local_dir, filename, use_symlinks=False, repo_type=None):
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
    print("-" * 60)
    os.makedirs(local_dir, exist_ok=True)

    try:
        print(f"Downloading file:")
        print(f"  Repo: {repo_id}")
        print(f"  File: {filename}")
        print(f"  To:   {local_dir}")
        print(f"  Symlinks: {use_symlinks}")
        print(f"  Repo Type: {repo_type or 'default'}")
        
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=use_symlinks,
            resume_download=True,
            repo_type=repo_type
        )
        print(f"Successfully downloaded: {file_path}")
        print("-" * 60)
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

    print("-" * 60)
    return False

def main():
    """Downloads Flux GGUF models and face segmentation model based on user VRAM selection."""
    print("Starting Flux GGUF model downloads...")
    print(f"Base download directory: {os.path.abspath(BASE_DOWNLOAD_DIR)}")
    print(f"Symlinks: {'Enabled' if USE_SYMLINKS else 'Disabled (copying files)'}")

    # Get user's VRAM and model choice
    unet_filename, quant_level = get_user_vram_choice()
    t5_filename = T5_ENCODER_MODELS.get(quant_level, "t5-v1_1-xxl-encoder-Q4_K_M.gguf")  # Fallback if quant not found

    # Define download tasks
    tasks = [
        {
            "repo_id": "QuantStack/FLUX.1-Kontext-dev-GGUF",
            "repo_type": "model",
            "filename": unet_filename,
            "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
        },
        {
            "repo_id": "city96/t5-v1_1-xxl-encoder-gguf",
            "repo_type": "model",
            "filename": t5_filename,
            "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip")
        },
        {
            "repo_id": "24xx/segm",
            "repo_type": "model",
            "filename": "face_yolov8n-seg2_60.pt",
            "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ultralytics", "segm")
        }
    ]

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

        success = download_file(
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

    print("\n" + "=" * 50)
    print("--- Download Summary ---")
    print(f"Selected flux model: {unet_filename} (Quant: {quant_level})")
    print(f"Selected t5 encoder model: {t5_filename}")
    print(f"Face segmentation model: face_yolov8n-seg2_60.pt")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print("=" * 50)

    if failed_downloads > 0:
        print("\nWarning: One or more downloads failed. Check error messages above.")
        sys.exit(1)
    else:
        print("\nAll downloads completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()