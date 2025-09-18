@echo off
setlocal EnableDelayedExpansion

:: Always run from this script's folder
pushd "%~dp0"

:: ───── Check Python & pip ─────
where python >nul 2>&1 || (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python and ensure it's added to your PATH.
    pause & exit /b 1
)

where pip >nul 2>&1 || (
    echo [ERROR] pip is not installed or not in PATH.
    echo Please ensure pip is installed with Python.
    pause & exit /b 1
)

:: ───── Install required packages (skip if already up‑to‑date) ─────
echo [INFO] Installing/upgrading required Python packages...
pip install --quiet --upgrade hf_transfer                          || goto :pipfail
pip install --quiet --upgrade "huggingface_hub[cli,tqdm]"          || goto :pipfail
pip install --quiet --upgrade "huggingface_hub[hf_xet]"            || goto :pipfail
echo [SUCCESS] Packages installed successfully.
echo.

:menu
cls
echo ================================================
echo        AI MODEL DOWNLOADER LAUNCHER
echo ================================================
echo.
echo Choose a model to download:
echo.
echo 1. Flux Dev FP8        (24 GB or more VRAM)
echo 2. Flux Dev GGUF       (24 GB or lower VRAM)
echo 3. Flux Kontext GGUF   (24 GB or lower VRAM)
echo 4. Wan 2.1 GGUF InfiniteTalk (32 GB VRAM or Less)
echo 5. Wan 2.1 Vace GGUF   (32 GB VRAM or Less)
echo 6. Wan 2.1 Phantom GGUF (32 GB VRAM or Less)
echo 7. NSFW Lesson Models (Adult Content)
echo 8. Wan 2.2 T2V Models (Text-to-Video)
echo 9. Wan 2.2 I2V Models (Image-to-Video)
echo 10. Exit
echo.
echo ================================================
set /p choice="Enter your choice (1-9): "

:: Validate input
if "%choice%"=="" (
    echo [ERROR] No choice entered. Please try again.
    pause
    goto menu
)

:: Process choice
if "%choice%"=="1" (
    echo [INFO] Launching Flux Dev FP8 downloader...
    if exist "Download_fluxDev_models_FP8.py" (
        python "Download_fluxDev_models_FP8.py"
    ) else (
        echo [ERROR] Download_fluxDev_models_FP8.py not found in current directory.
        pause
    )
) else if "%choice%"=="2" (
    echo [INFO] Launching Flux Dev GGUF downloader...
    if exist "Download_fluxDev_models_GGUF.py" (
        python "Download_fluxDev_models_GGUF.py"
    ) else (
        echo [ERROR] Download_fluxDev_models_GGUF.py not found in current directory.
        pause
    )
) else if "%choice%"=="3" (
    echo [INFO] Launching Flux Kontext GGUF downloader...
    if exist "Download_models_Flux_Kontext_GGUF.py" (
        python "Download_models_Flux_Kontext_GGUF.py"
    ) else (
        echo [ERROR] Download_models_Flux_Kontext_GGUF.py not found in current directory.
        pause
    )
) else if "%choice%"=="4" (
    echo [INFO] Launching Wan 2.1 GGUF downloader...
    if exist "Download_models_GGUF.py" (
        python "Download_models_GGUF.py"
    ) else (
        echo [ERROR] Download_models_GGUF.py not found in current directory.
        pause
    )
) else if "%choice%"=="5" (
    echo [INFO] Launching Wan 2.1 Vace GGUF downloader...
    if exist "Download_models_GGUF_VACE.py" (
        python "Download_models_GGUF_VACE.py"
    ) else (
        echo [ERROR] Download_models_GGUF_VACE.py not found in current directory.
        pause
    )
) else if "%choice%"=="6" (
    echo [INFO] Launching Wan 2.1 Phantom GGUF downloader...
    if exist "Download_models_GGUF_PHANTOM.py" (
        python "Download_models_GGUF_PHANTOM.py"
    ) else (
        echo [ERROR] Download_models_GGUF_PHANTOM.py not found in current directory.
        pause
    )
) else if "%choice%"=="7" (
    echo [INFO] Launching NSFW Lessons Models downloader...
    if exist "Download_models_NSFW.py" (
        python "Download_models_NSFW.py"
    ) else (
        echo [ERROR] Download_models_NSFW.py not found in current directory.
        pause
    )
) else if "%choice%"=="8" (
    echo [INFO] Launching Wan 2.2 T2V Models downloader...
    if exist "Download_wan2-2_T2V.py" (
        python "Download_wan2-2_T2V.py"
    ) else (
        echo [ERROR] Download_wan2-2_T2V.py not found in current directory.
        pause
	)
) else if "%choice%"=="9" (
    echo [INFO] Launching Wan 2.2 I2V Models downloader...
    if exist "Download_wan2-2_I2V.py" (
        python "Download_wan2-2_I2V.py"
    ) else (
        echo [ERROR] Download_wan2-2_I2V.py not found in current directory.
        pause
	)
) else if "%choice%"=="10" (
    echo [INFO] Exiting...
    echo Thank you for using the AI Model Downloader!
    pause
    goto :eof
) else (
    echo [ERROR] Invalid choice "%choice%". Please enter a number between 1-9.
    pause
    goto menu
)

echo.
echo [INFO] Download process completed.
pause
goto menu

:pipfail
echo [ERROR] One of the package installations failed. 
echo Please check the terminal output above for details.
echo.
echo Common solutions:
echo - Ensure you have internet connection
echo - Try running this script as administrator
echo - Update pip: python -m pip install --upgrade pip
pause
exit /b 1