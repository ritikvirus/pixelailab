# --- START OF Download_fluxDev_models(FP8).py ---

import os
import sys
import zipfile
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
USE_SYMLINKS = False  # Copy files instead of symlinking to ensure compatibility

# Define download tasks for individual files and ZIP files
DOWNLOAD_TASKS = [
    # --- unet ---
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",  # Specify dataset type
        "filename": "flux1-fill-dev-fp8.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
    },
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "flux1-dev-fp8.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
    },
    # --- controlnet ---
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "Flux-Union-Pro2.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "controlnet")
    },
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "Flux1-controlnet-upscaler-Jasperai-fp8.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "controlnet")
    },
    # --- clip ---
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "clip_l.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip")
    },
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip")
    },
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "t5xxl_fp8_e4m3fn_scaled.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip")
    },
    # --- vae ---
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "ae.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "vae")
    },
    # --- loras ---
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "comfyui_portrait_lora64.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "Maya_Lora_v1_000002500.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "more_details.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
        {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "FameGrid_Bold_SDXL_V1.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
    # --- checkpoints ---
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "epicrealism_naturalSinRC1VAE.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "checkpoints")
    },
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "epicrealism_pureEvolutionV3.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "checkpoints")
    },
    {
        "repo_id": "simwalo/FluxDevFP8",
        "repo_type": "dataset",
        "filename": "epicrealismXL_vxviLastfameRealism.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "checkpoints")
    },

        # --- Upscalers ---
     {
        "repo_id": "simwalo/Wan2.1_SkyreelsV2",
        "repo_type": "dataset",
        "filename": "RealESRGAN_x2plus.pth",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "upscale_models")
    },
    # --- ZIP files (to be extracted and deleted) ---
    {
        "repo_id": "simwalo/custom_nodes",
        "repo_type": "dataset",
        "filename": "PersonMaskUltraV2.zip",
        "local_dir": BASE_DOWNLOAD_DIR,
        "extract_and_delete": True
    },
    {
        "repo_id": "simwalo/custom_nodes",
        "repo_type": "dataset",
        "filename": "insightface.zip",
        "local_dir": BASE_DOWNLOAD_DIR,
        "extract_and_delete": True
    },
    {
        "repo_id": "simwalo/custom_nodes",
        "repo_type": "dataset",
        "filename": "liveportrait.zip",
        "local_dir": BASE_DOWNLOAD_DIR,
        "extract_and_delete": True
    },
    {
        "repo_id": "simwalo/custom_nodes",
        "repo_type": "dataset",
        "filename": "rembg.zip",
        "local_dir": BASE_DOWNLOAD_DIR,
        "extract_and_delete": True
    },
 {
        "repo_id": "simwalo/custom_nodes",
        "repo_type": "dataset",
        "filename": "LLM.zip",
        "local_dir": BASE_DOWNLOAD_DIR,
        "extract_and_delete": True
    },
    {
    "repo_id": "simwalo/custom_nodes",
    "repo_type": "dataset",
    "filename": "sams.zip",
    "local_dir": BASE_DOWNLOAD_DIR,
    "extract_and_delete": True
},

   # --- New task for ComfyUI-LatentSyncWrapper.zip ---
    {
        "repo_id": "simwalo/custom_nodes",
        "repo_type": "dataset",
        "filename": "ComfyUI-LatentSyncWrapper.zip",
        "local_dir": "ComfyUI/custom_nodes",
        "extract_and_delete": True
    },
        {
        "repo_id": "simwalo/custom_nodes",
        "repo_type": "dataset",
        "filename": "comfyui-reactor.zip",
        "local_dir": "ComfyUI/custom_nodes",
        "extract_and_delete": True
    },
        {
        "repo_id": "simwalo/custom_nodes",
        "repo_type": "dataset",
        "filename": "Joy_caption_two.zip",
        "local_dir": "ComfyUI/models",
        "extract_and_delete": True
    }
    
]

def download_and_process_item(repo_id, local_dir, filename, use_symlinks=False, extract_and_delete=False, repo_type=None):
    """
    Downloads a file from a Hugging Face repository and optionally extracts and deletes ZIP files.

    Args:
        repo_id (str): Hugging Face repository ID.
        local_dir (str): Target directory for the file.
        filename (str): Specific file to download.
        use_symlinks (bool): Controls symlink behavior.
        extract_and_delete (bool): If True, extract ZIP file and delete it after extraction.
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

        if extract_and_delete and filename.lower().endswith('.zip'):
            print(f"Extracting ZIP file: {file_path}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(local_dir)
            print(f"Extracted contents to: {local_dir}")

            print(f"Deleting ZIP file: {file_path}")
            os.remove(file_path)
            print(f"Deleted ZIP file: {file_path}")

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
    except zipfile.BadZipFile:
        print(f"\nError: File '{filename}' is not a valid ZIP file.", file=sys.stderr)
    except Exception as e:
        print(f"\nError: Unexpected error downloading '{filename}' from '{repo_id}': {type(e).__name__} - {e}", file=sys.stderr)

    print("-" * 20)
    return False

def main():
    """Downloads specified FluxDevFP8 models and custom nodes, handling ZIP extraction."""
    print("Starting FluxDevFP8 model and custom node downloads...")
    print(f"Base download directory: {os.path.abspath(BASE_DOWNLOAD_DIR)}")
    print(f"Symlinks: {'Enabled' if USE_SYMLINKS else 'Disabled (copying files)'}")

    successful_downloads = 0
    failed_downloads = 0

    for i, task in enumerate(DOWNLOAD_TASKS, 1):
        print(f"\n--- Task {i}/{len(DOWNLOAD_TASKS)} ---")
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
            extract_and_delete=task.get("extract_and_delete", False),
            repo_type=repo_type
        )
        if success:
            successful_downloads += 1
        else:
            failed_downloads += 1
            print(f"Continuing with next task despite error...")

    print("\n" + "=" * 30)
    print("--- Download Summary ---")
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

# --- END OF Download_fluxDev_models(FP8).py ---