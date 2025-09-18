import os
import sys
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError
)

# --- Configuration ---
BASE_DOWNLOAD_DIR = "ComfyUI/models"
USE_SYMLINKS = False  # Copy files instead of symlinking to ensure compatibility
MAX_CONCURRENT_DOWNLOADS = 6  # Adjust based on your internet speed and system resources

# Thread-safe print function
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

# Define download tasks for all SDXL + IPAdapter models
DOWNLOAD_TASKS = [
    # --- Checkpoints ---
    {
        "repo_id": "simwalo/SDXL",
        "repo_type": "model",
        "filename": "bigLust_v16.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "checkpoints")
    },
    {
        "repo_id": "simwalo/SDXL",
        "repo_type": "model",
        "filename": "analogMadnessSDXL_xl2.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "checkpoints")
    },
    
    
    # --- IPAdapter CLIP Vision Models ---
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "models/image_encoder/model.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip_vision"),
        "rename_to": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "sdxl_models/image_encoder/model.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip_vision"),
        "rename_to": "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"
    },
    {
        "repo_id": "Kwai-Kolors/Kolors-IP-Adapter-Plus",
        "repo_type": "model",
        "filename": "image_encoder/pytorch_model.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip_vision"),
        "rename_to": "clip-vit-large-patch14-336.bin"
    },
    
    # --- IPAdapter Models ---
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "models/ip-adapter_sd15.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "models/ip-adapter_sd15_light_v11.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "models/ip-adapter-plus_sd15.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "models/ip-adapter-plus-face_sd15.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "models/ip-adapter-full-face_sd15.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "models/ip-adapter_sd15_vit-G.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "sdxl_models/ip-adapter_sdxl.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter",
        "repo_type": "model",
        "filename": "models/ip-adapter_sd15_light.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    
    # --- IPAdapter FaceID Models ---
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid_sd15.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid-plusv2_sd15.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid-portrait-v11_sd15.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid_sdxl.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid-plusv2_sdxl.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid-portrait_sdxl.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid-portrait_sdxl_unnorm.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid-plus_sd15.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid-portrait_sd15.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    
    # --- IPAdapter FaceID LoRA Models ---
    {
        "repo_id": "simwalo/SDXL",
        "repo_type": "model",
        "filename": "Touch_of_Realism_SDXL_V2.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid_sd15_lora.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid-plusv2_sd15_lora.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid_sdxl_lora.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
    {
        "repo_id": "h94/IP-Adapter-FaceID",
        "repo_type": "model",
        "filename": "ip-adapter-faceid-plus_sd15_lora.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
    },
    
    # --- IPAdapter Community Models ---
    {
        "repo_id": "ostris/ip-composition-adapter",
        "repo_type": "model",
        "filename": "ip_plus_composition_sd15.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "ostris/ip-composition-adapter",
        "repo_type": "model",
        "filename": "ip_plus_composition_sdxl.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter")
    },
    {
        "repo_id": "Kwai-Kolors/Kolors-IP-Adapter-Plus",
        "repo_type": "model",
        "filename": "ip_adapter_plus_general.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter"),
        "rename_to": "Kolors-IP-Adapter-Plus.bin"
    },
    {
        "repo_id": "Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus",
        "repo_type": "model",
        "filename": "ipa-faceid-plus.bin",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ipadapter"),
        "rename_to": "Kolors-IP-Adapter-FaceID-Plus.bin"
    },

    # --- ControlNet Models ---
    {
        "repo_id": "simwalo/SDXL",
        "repo_type": "model",
        "filename": "Depth-SDXL-xinsir.safetensors",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "controlnet")
    },

    # --- Upscaler Models ---
    {
        "repo_id": "lokCX/4x-Ultrasharp",
        "repo_type": "model",
        "filename": "4x-UltraSharp.pth",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "upscale_models")
    },
    {
        "repo_id": "skbhadra/ClearRealityV1",
        "repo_type": "model",
        "filename": "4x-ClearRealityV1.pth",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "upscale_models")
    },

    # --- Segmentation Models ---
    {
        "repo_id": "24xx/segm",
        "repo_type": "model",
        "filename": "face_yolov8n-seg2_60.pt",
        "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "ultralytics", "segm")
    },
    
    # --- Custom Nodes (ZIP files to be extracted and deleted) ---
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
        "filename": "Joy_caption_two.zip",
        "local_dir": "ComfyUI/models",
        "extract_and_delete": True
    },
    {
        "repo_id": "simwalo/SDXL",
        "repo_type": "model",
        "filename": "comfyui_controlnet_aux.zip",
        "local_dir": "ComfyUI/custom_nodes",
        "extract_and_delete": True
    },
    {
        "repo_id": "simwalo/SDXL",
        "repo_type": "model",
        "filename": "Text_Processor_By_Aiconomist.zip",
        "local_dir": "ComfyUI/custom_nodes",
        "extract_and_delete": True
    }
]

def download_and_process_item(repo_id, local_dir, filename, use_symlinks=False, extract_and_delete=False, repo_type=None, rename_to=None):
    """
    Downloads a file from a Hugging Face repository and optionally extracts and deletes ZIP files.

    Args:
        repo_id (str): Hugging Face repository ID.
        local_dir (str): Target directory for the file.
        filename (str): Specific file to download.
        use_symlinks (bool): Controls symlink behavior.
        extract_and_delete (bool): If True, extract ZIP file and delete it after extraction.
        repo_type (str): Type of repository ('dataset', 'model', etc.).
        rename_to (str): Optional new filename after download.

    Returns:
        bool: True if successful, False otherwise.
    """
    safe_print("-" * 50)
    os.makedirs(local_dir, exist_ok=True)

    try:
        display_name = rename_to if rename_to else os.path.basename(filename)
        
        # Check if file already exists
        final_path = os.path.join(local_dir, display_name)
        if os.path.exists(final_path) and not extract_and_delete:
            safe_print(f"⏭️  File already exists, skipping: {display_name}")
            return True

        safe_print(f"🔄 Downloading:\n  📁 Repo: {repo_id}\n  📄 File: {filename}\n  📂 To: {local_dir}\n  🔗 Symlinks: {use_symlinks}\n  🏷️  Type: {repo_type or 'default'}")
        
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=use_symlinks,
            resume_download=True,
            repo_type=repo_type
        )
        
        # Handle renaming if required
        if rename_to:
            original_path = file_path
            new_path = os.path.join(local_dir, rename_to)
            
            if os.path.basename(original_path) != rename_to:
                if os.path.exists(new_path):
                    os.remove(original_path)
                else:
                    os.rename(original_path, new_path)
                file_path = new_path
        
        safe_print(f"✅ Successfully downloaded: {file_path}")

        if extract_and_delete and filename.lower().endswith('.zip'):
            safe_print(f"🗜️  Extracting ZIP file: {file_path}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(local_dir)
            safe_print(f"📦 Extracted contents to: {local_dir}")

            safe_print(f"🗑️  Deleting ZIP file: {file_path}")
            os.remove(file_path)
            safe_print(f"✅ Deleted ZIP file: {file_path}")

        safe_print("-" * 50)
        return True

    except RepositoryNotFoundError as e:
        safe_print(f"❌ Error: Repository not found for '{repo_id}'. Details: {e}")
    except EntryNotFoundError:
        safe_print(f"❌ Error: File '{filename}' not found in repo '{repo_id}'.")
    except LocalEntryNotFoundError as e:
        safe_print(f"❌ Error: Local file system issue for '{repo_id}' in '{local_dir}'. Details: {e}")
    except HfHubHTTPError as e:
        safe_print(f"❌ Error: HTTP error for repo '{repo_id}'. Status: {e.response.status_code}. Details: {e}")
    except zipfile.BadZipFile:
        safe_print(f"❌ Error: File '{filename}' is not a valid ZIP file.")
    except Exception as e:
        safe_print(f"❌ Error: Unexpected error downloading '{filename}' from '{repo_id}': {type(e).__name__} - {e}")

    safe_print("-" * 50)
    return False

def worker_thread(task_queue, results_queue):
    """Worker thread function for processing download tasks"""
    while True:
        task_data = task_queue.get()
        if task_data is None:
            break
        
        task_num, task = task_data
        repo_id = task.get("repo_id")
        local_dir = task.get("local_dir")
        filename = task.get("filename")
        repo_type = task.get("repo_type")
        rename_to = task.get("rename_to")

        if not all([repo_id, local_dir, filename]):
            safe_print(f"❌ Error: Task {task_num} is missing required fields. Skipping.")
            results_queue.put((task_num, False))
            task_queue.task_done()
            continue

        success = download_and_process_item(
            repo_id=repo_id,
            local_dir=local_dir,
            filename=filename,
            use_symlinks=USE_SYMLINKS,
            extract_and_delete=task.get("extract_and_delete", False),
            repo_type=repo_type,
            rename_to=rename_to
        )
        
        results_queue.put((task_num, success))
        task_queue.task_done()

def main():
    """Downloads specified SDXL + IPAdapter models and custom nodes with parallel processing."""
    print("=" * 80)
    print("🚀 PARALLEL SDXL + IPADAPTER MODEL DOWNLOADER")
    print("=" * 80)
    print(f"📁 Base download directory: {os.path.abspath(BASE_DOWNLOAD_DIR)}")
    print(f"🔗 Symlinks: {'Enabled' if USE_SYMLINKS else 'Disabled (copying files)'}")
    print(f"⚡ Max concurrent downloads: {MAX_CONCURRENT_DOWNLOADS}")
    print(f"📦 Total download tasks: {len(DOWNLOAD_TASKS)}")
    print("=" * 80)

    # Create required directories
    required_dirs = [
        os.path.join(BASE_DOWNLOAD_DIR, "checkpoints"),
        os.path.join(BASE_DOWNLOAD_DIR, "unet"),
        os.path.join(BASE_DOWNLOAD_DIR, "clip"),
        os.path.join(BASE_DOWNLOAD_DIR, "clip_vision"),
        os.path.join(BASE_DOWNLOAD_DIR, "vae"),
        os.path.join(BASE_DOWNLOAD_DIR, "ipadapter"),
        os.path.join(BASE_DOWNLOAD_DIR, "loras"),
        os.path.join(BASE_DOWNLOAD_DIR, "controlnet"),
        os.path.join(BASE_DOWNLOAD_DIR, "upscale_models"),
        os.path.join(BASE_DOWNLOAD_DIR, "ultralytics", "segm"),
        "ComfyUI/custom_nodes"
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("📁 Created required directories")

    successful_downloads = 0
    failed_downloads = 0
    skipped_downloads = 0
    start_time = time.time()

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        # Submit all tasks
        future_to_task = {}
        for i, task in enumerate(DOWNLOAD_TASKS, 1):
            future = executor.submit(
                download_and_process_item,
                task.get("repo_id"),
                task.get("local_dir"),
                task.get("filename"),
                USE_SYMLINKS,
                task.get("extract_and_delete", False),
                task.get("repo_type"),
                task.get("rename_to")
            )
            future_to_task[future] = (i, task)

        # Process completed tasks
        for future in as_completed(future_to_task):
            task_num, task = future_to_task[future]
            try:
                success = future.result()
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
                    safe_print(f"⚠️  Task {task_num} failed, continuing with next task...")
            except Exception as e:
                safe_print(f"❌ Task {task_num} encountered an exception: {e}")
                failed_downloads += 1

    # Calculate final statistics
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📦 Total tasks processed: {len(DOWNLOAD_TASKS)}")
    print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    if successful_downloads > 0:
        print(f"🚀 Average time per successful download: {total_time/successful_downloads:.2f} seconds")
    print("=" * 80)

    if failed_downloads > 0:
        print(f"\n⚠️  Warning: {failed_downloads} task(s) failed. Check error messages above.")
        print("💡 TIP: You can re-run the script to retry failed downloads")
        sys.exit(1)
    else:
        print("\n🎉 ALL DOWNLOADS COMPLETED SUCCESSFULLY!")
        print(f"📁 Files downloaded to: {os.path.abspath(BASE_DOWNLOAD_DIR)}")
        sys.exit(0)

if __name__ == "__main__":
    main()