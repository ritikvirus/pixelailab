import os
import sys
import zipfile
import asyncio
import concurrent.futures
from threading import Lock
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
MAX_CONCURRENT_DOWNLOADS = 4  # Adjust based on your bandwidth and system capabilities

# Thread-safe counters
download_lock = Lock()
successful_downloads = 0
failed_downloads = 0

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

def update_counters(success):
    """Thread-safe counter updates"""
    global successful_downloads, failed_downloads
    with download_lock:
        if success:
            successful_downloads += 1
        else:
            failed_downloads += 1

def download_and_process_item(task_info):
    """
    Downloads a file from a Hugging Face repository and optionally extracts and deletes ZIP files.
    
    Args:
        task_info (tuple): (task_index, task_dict) containing task information
    
    Returns:
        tuple: (task_index, success_bool, task_dict)
    """
    task_index, task = task_info
    repo_id = task.get("repo_id")
    local_dir = task.get("local_dir")
    filename = task.get("filename")
    repo_type = task.get("repo_type")
    extract_and_delete = task.get("extract_and_delete", False)
    
    print(f"[Task {task_index + 1}] Starting download: {filename}")
    
    if not all([repo_id, local_dir, filename]):
        print(f"[Task {task_index + 1}] Error: Missing required fields. Skipping.", file=sys.stderr)
        update_counters(False)
        return (task_index, False, task)

    os.makedirs(local_dir, exist_ok=True)

    try:
        print(f"[Task {task_index + 1}] Downloading:\n  Repo: {repo_id}\n  File: {filename}\n  To: {local_dir}")
        
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=USE_SYMLINKS,
            resume_download=True,
            repo_type=repo_type
            # token="your_hf_token_here"  # Uncomment and add token if repository is private/gated
        )
        
        print(f"[Task {task_index + 1}] Successfully downloaded: {file_path}")

        if extract_and_delete and filename.lower().endswith('.zip'):
            print(f"[Task {task_index + 1}] Extracting ZIP file: {file_path}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(local_dir)
            print(f"[Task {task_index + 1}] Extracted contents to: {local_dir}")

            print(f"[Task {task_index + 1}] Deleting ZIP file: {file_path}")
            os.remove(file_path)
            print(f"[Task {task_index + 1}] Deleted ZIP file: {file_path}")

        update_counters(True)
        return (task_index, True, task)

    except RepositoryNotFoundError as e:
        print(f"[Task {task_index + 1}] Error: Repository not found for '{repo_id}'. Details: {e}", file=sys.stderr)
    except EntryNotFoundError:
        print(f"[Task {task_index + 1}] Error: File '{filename}' not found in repo '{repo_id}'.", file=sys.stderr)
    except LocalEntryNotFoundError as e:
        print(f"[Task {task_index + 1}] Error: Local file system issue for '{repo_id}' in '{local_dir}'. Details: {e}", file=sys.stderr)
    except HfHubHTTPError as e:
        print(f"[Task {task_index + 1}] Error: HTTP error for repo '{repo_id}'. Status: {e.response.status_code}. Details: {e}", file=sys.stderr)
    except zipfile.BadZipFile:
        print(f"[Task {task_index + 1}] Error: File '{filename}' is not a valid ZIP file.", file=sys.stderr)
    except Exception as e:
        print(f"[Task {task_index + 1}] Error: Unexpected error downloading '{filename}' from '{repo_id}': {type(e).__name__} - {e}", file=sys.stderr)

    update_counters(False)
    return (task_index, False, task)

def main():
    """Downloads specified FluxDevFP8 models and custom nodes in parallel, handling ZIP extraction."""
    print("Starting FluxDevFP8 model and custom node downloads (PARALLEL MODE)...")
    print(f"Base download directory: {os.path.abspath(BASE_DOWNLOAD_DIR)}")
    print(f"Symlinks: {'Enabled' if USE_SYMLINKS else 'Disabled (copying files)'}")
    print(f"Max concurrent downloads: {MAX_CONCURRENT_DOWNLOADS}")
    print(f"Total tasks: {len(DOWNLOAD_TASKS)}")
    print("=" * 50)

    # Prepare tasks with indices for tracking
    indexed_tasks = list(enumerate(DOWNLOAD_TASKS))
    
    # Use ThreadPoolExecutor for parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(download_and_process_item, task): task for task in indexed_tasks}
        
        # Process completed tasks as they finish
        completed_tasks = []
        for future in concurrent.futures.as_completed(future_to_task):
            task_index, success, task = future.result()
            completed_tasks.append((task_index, success, task))
            
            # Print progress
            progress = len(completed_tasks)
            print(f"\n--- Progress: {progress}/{len(DOWNLOAD_TASKS)} tasks completed ---")
            
            if not success:
                print(f"Task {task_index + 1} failed, but continuing with remaining tasks...")

    print("\n" + "=" * 50)
    print("--- Download Summary ---")
    print(f"Total tasks: {len(DOWNLOAD_TASKS)}")
    print(f"Successful tasks: {successful_downloads}")
    print(f"Failed tasks: {failed_downloads}")
    print("=" * 50)

    # Show failed tasks if any
    if failed_downloads > 0:
        print("\nFailed tasks:")
        failed_tasks = [task for _, success, task in completed_tasks if not success]
        for i, task in enumerate(failed_tasks, 1):
            print(f"  {i}. {task.get('filename', 'Unknown')} from {task.get('repo_id', 'Unknown repo')}")
        
        print(f"\nWarning: {failed_downloads} task(s) failed. Check error messages above.")
        sys.exit(1)
    else:
        print("\nAll tasks completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()