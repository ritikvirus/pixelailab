import os
import sys
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
MAX_CONCURRENT_DOWNLOADS = 2  # Reduced due to large file sizes

# Thread-safe print function
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def get_user_choice():
    """Get user's choice for model type"""
    print("=" * 80)
    print("🚀 WAN2.2 I2V MODEL DOWNLOADER")
    print("=" * 80)
    print("Choose which model type to download:")
    print()
    print("1. GGUF Models (Recommended for 12GB+ VRAM)")
    print("   - Highly quantized models with smallest file sizes")
    print("   - Best memory efficiency for lower-end GPUs")
    print("   - Suitable for most consumer GPUs (8-12GB VRAM)")
    print()
    print("2. FP8 Models (Recommended for 16GB+ VRAM)")
    print("   - Quantized models with smaller file sizes (~14.3GB each)")
    print("   - Better memory efficiency")
    print("   - Suitable for most high-end consumer GPUs")
    print()
    print("3. FP16 Models (Recommended for 32GB+ VRAM)")
    print("   - Higher precision models (~28.6GB each)")
    print("   - Better quality but much larger file sizes")
    print("   - Requires professional/workstation GPUs")
    print()
    print("4. Exit")
    print("=" * 80)
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n\n👋 Download cancelled by user.")
            sys.exit(0)
        except EOFError:
            print("\n\n❌ Input error. Exiting.")
            sys.exit(1)

def get_download_tasks(model_type):
    """Get download tasks based on model type selection"""
    
    # Common models for both types
    common_tasks = [
        # --- VAE Models (same for both) ---
        {
            "repo_id": "QuantStack/Wan2.2-T2V-A14B-GGUF",
            "repo_type": "model",
            "filename": "VAE/Wan2.1_VAE.safetensors",
            "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "vae")
        },
        
        # --- LoRA Models (same for both) ---
        {
            "repo_id": "simwalo/Wan2.1_SkyreelsV2",
            "repo_type": "dataset",
            "filename": "Instagirlv2.0_hinoise.safetensors",
            "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
        },
        {
            "repo_id": "simwalo/Wan2.1_SkyreelsV2",
            "repo_type": "dataset",
            "filename": "Instagirlv2.0_lownoise.safetensors",
            "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
        },
        {
            "repo_id": "Kijai/WanVideo_comfy",
            "repo_type": "model",
            "filename": "Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
            "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
        },
        {
            "repo_id": "Kijai/WanVideo_comfy",
            "repo_type": "model",
            "filename": "Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
            "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "loras")
        }
    ]
    
    if model_type == "gguf":
        # GGUF specific models
        gguf_tasks = [
            # --- UNET Models (GGUF) ---
            {
                "repo_id": "QuantStack/Wan2.2-I2V-A14B-GGUF",
                "repo_type": "model",
                "filename": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
                "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
            },
            {
                "repo_id": "QuantStack/Wan2.2-I2V-A14B-GGUF",
                "repo_type": "model",
                "filename": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
                "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
            },
            
            # --- CLIP Models (GGUF) ---
            {
                "repo_id": "city96/umt5-xxl-encoder-gguf",
                "repo_type": "model",
                "filename": "umt5-xxl-encoder-Q4_K_M.gguf",
                "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip")
            }
        ]
        return common_tasks + gguf_tasks
        
    elif model_type == "fp8":
        # FP8 specific models
        fp8_tasks = [
            # --- UNET Models (FP8) ---
            {
                "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "repo_type": "model",
                "filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
            },
            {
                "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "repo_type": "model",
                "filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
            },
            
            # --- CLIP Models (FP8) ---
            {
                "repo_id": "Kijai/WanVideo_comfy",
                "repo_type": "model",
                "filename": "umt5-xxl-enc-fp8_e4m3fn.safetensors",
                "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip")
            }
        ]
        return common_tasks + fp8_tasks
        
    elif model_type == "fp16":
        # FP16 specific models
        fp16_tasks = [
            # --- UNET Models (FP16) ---
            {
                "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "repo_type": "model",
                "filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
                "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
            },
            {
                "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "repo_type": "model",
                "filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
                "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "unet")
            },
            
            # --- CLIP Models (FP16 - using compatible model) ---
            {
                "repo_id": "Kijai/WanVideo_comfy",
                "repo_type": "model",
                "filename": "umt5-xxl-enc-fp8_e4m3fn.safetensors",
                "local_dir": os.path.join(BASE_DOWNLOAD_DIR, "clip")
            }
        ]
        return common_tasks + fp16_tasks

def download_and_process_item(repo_id, local_dir, filename, use_symlinks=False, repo_type=None, rename_to=None):
    """
    Downloads a file from a Hugging Face repository.

    Args:
        repo_id (str): Hugging Face repository ID.
        local_dir (str): Target directory for the file.
        filename (str): Specific file to download.
        use_symlinks (bool): Controls symlink behavior.
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
        if os.path.exists(final_path):
            safe_print(f"⭐️ File already exists, skipping: {display_name}")
            return True

        safe_print(f"📥 Downloading:\n  📁 Repo: {repo_id}\n  📄 File: {filename}\n  📂 To: {local_dir}\n  🔗 Symlinks: {use_symlinks}\n  🏷️ Type: {repo_type or 'default'}")
        
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
    except Exception as e:
        safe_print(f"❌ Error: Unexpected error downloading '{filename}' from '{repo_id}': {type(e).__name__} - {e}")

    safe_print("-" * 50)
    return False

def main():
    """Downloads specified Wan2.2 I2V models with parallel processing."""
    
    # Get user choice
    choice = get_user_choice()
    
    if choice == '4':
        print("\n👋 Exiting downloader. Have a great day!")
        sys.exit(0)
    
    # Determine model type and get tasks
    if choice == '1':
        model_type = "gguf"
        model_name = "GGUF (8GB+ VRAM)"
    elif choice == '2':
        model_type = "fp8"
        model_name = "FP8 (16GB+ VRAM)"
    else:  # choice == '3'
        model_type = "fp16"
        model_name = "FP16 (32GB+ VRAM)"
    
    download_tasks = get_download_tasks(model_type)
    
    print(f"\n🎯 Selected: {model_name}")
    print("=" * 80)
    print(f"📁 Base download directory: {os.path.abspath(BASE_DOWNLOAD_DIR)}")
    print(f"🔗 Symlinks: {'Enabled' if USE_SYMLINKS else 'Disabled (copying files)'}")
    print(f"⚡ Max concurrent downloads: {MAX_CONCURRENT_DOWNLOADS}")
    print(f"📦 Total download tasks: {len(download_tasks)}")
    
    if model_type == "fp8":
        print("💾 Expected download size: ~28.6GB (2x 14.3GB models)")
    elif model_type == "fp16":
        print("💾 Expected download size: ~57.2GB (2x 28.6GB models)")
    else:  # gguf
        print("💾 Expected download size: ~6-8GB (GGUF quantized models)")
    
    print("=" * 80)
    print("⚠️  WARNING: These are large files. Ensure you have sufficient:")
    if model_type == "gguf":
        print("   • Disk space (15GB+ recommended)")
        print("   • Stable internet connection") 
        print("   • Time for download (30-60 minutes)")
    else:
        print("   • Disk space (60GB+ recommended)")
        print("   • Stable internet connection")
        print("   • Time for download (may take hours)")
    print("=" * 80)
    
    # Confirmation for large downloads
    confirm = input("Do you want to continue with the download? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("\n👋 Download cancelled by user.")
        sys.exit(0)

    # Create required directories
    required_dirs = [
        os.path.join(BASE_DOWNLOAD_DIR, "unet"),
        os.path.join(BASE_DOWNLOAD_DIR, "vae"),
        os.path.join(BASE_DOWNLOAD_DIR, "clip"),
        os.path.join(BASE_DOWNLOAD_DIR, "loras")
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("📁 Created required directories")

    successful_downloads = 0
    failed_downloads = 0
    start_time = time.time()

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        # Submit all tasks
        future_to_task = {}
        for i, task in enumerate(download_tasks, 1):
            future = executor.submit(
                download_and_process_item,
                task.get("repo_id"),
                task.get("local_dir"),
                task.get("filename"),
                USE_SYMLINKS,
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
                    safe_print(f"⚠️ Task {task_num} failed, continuing with next task...")
            except Exception as e:
                safe_print(f"❌ Task {task_num} encountered an exception: {e}")
                failed_downloads += 1

    # Calculate final statistics
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"🎯 Model Type: {model_name}")
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📦 Total tasks processed: {len(download_tasks)}")
    print(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    if successful_downloads > 0:
        print(f"🚀 Average time per successful download: {total_time/successful_downloads:.2f} seconds")
    print("=" * 80)

    if failed_downloads > 0:
        print(f"\n⚠️ Warning: {failed_downloads} task(s) failed. Check error messages above.")
        print("💡 TIP: You can re-run the script to retry failed downloads")
        sys.exit(1)
    else:
        print("\n🎉 ALL DOWNLOADS COMPLETED SUCCESSFULLY!")
        print(f"📁 Files downloaded to: {os.path.abspath(BASE_DOWNLOAD_DIR)}")
        
        if model_type == "gguf":
            print("\n📋 Downloaded GGUF I2V models:")
            print("  🧠 UNET Models:")
            print("    - Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf")
            print("    - Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf")
            print("  📝 CLIP Model:")
            print("    - umt5-xxl-encoder-Q4_K_M.gguf")
        elif model_type == "fp8":
            print("\n📋 Downloaded FP8 I2V models:")
            print("  🧠 UNET Models:")
            print("    - wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors (~14.3GB)")
            print("    - wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors (~14.3GB)")
            print("  📝 CLIP Model:")
            print("    - umt5-xxl-enc-fp8_e4m3fn.safetensors")
        else:
            print("\n📋 Downloaded FP16 I2V models:")
            print("  🧠 UNET Models:")
            print("    - wan2.2_i2v_high_noise_14B_fp16.safetensors (~28.6GB)")
            print("    - wan2.2_i2v_low_noise_14B_fp16.safetensors (~28.6GB)")
            print("  📝 CLIP Model:")
            print("    - umt5-xxl-encoder-f16.gguf")
        
        print("  🎨 VAE Model:")
        print("    - Wan2.1_VAE.safetensors")
        print("  🎯 LoRA Models:")
        print("    - Instagirlv2.0_hinoise.safetensors")
        print("    - Instagirlv2.0_lownoise.safetensors")
        print("    - Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors")
        print("\n🎬 Ready for Image-to-Video generation!")
        sys.exit(0)

if __name__ == "__main__":
    main()