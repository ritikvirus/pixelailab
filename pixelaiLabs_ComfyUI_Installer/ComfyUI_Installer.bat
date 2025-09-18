@echo off
setlocal enabledelayedexpansion

@REM ------------------------------------------------------------------------------------------------
@REM Installing ComfyUI and a Venv within it
echo Installing ComfyUI, installing a Venv within it - choose your Python for it

git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI

echo From Comfy github page: "Python 3.12 is supported but using 3.10 is recommended because some custom nodes and their dependencies might not support it yet."

@REM Step 1: Check for Python in PATH first
echo Checking if Python is in PATH...
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Python found in PATH!
    python --version
    @REM Get the full path to Python executable
    for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)"') do set "PYTHON_PATH=%%i"
    echo Found Python at: !PYTHON_PATH!
    
    @REM Set up the arrays for selection
    set "INDEX=1"
    set "PYTHON_PATHS[1]=!PYTHON_PATH!"
    for /f "delims=" %%j in ('echo !PYTHON_PATH!') do set "PYTHON_BASE[1]=%%~dpj"
    set "PYTHON_BASE[1]=!PYTHON_BASE[1]:~0,-1!"
    echo 1. !PYTHON_BASE[1]!
    goto :check_additional_locations
) else (
    echo Python not found in PATH. Checking common locations...
    set "INDEX=0"
)

@REM Step 2: Check common Python installation directories
:check_additional_locations
set "PYTHON_DIR=C:\Users\%USERNAME%\AppData\Local\Programs\Python"
if exist "%PYTHON_DIR%" (
    for /d %%D in ("%PYTHON_DIR%\Python*") do (
        if exist "%%D\python.exe" (
            @REM Check if this path is already in our list
            set "ALREADY_FOUND=0"
            for /L %%N in (1,1,!INDEX!) do (
                if "!PYTHON_PATHS[%%N]!"=="%%D\python.exe" set "ALREADY_FOUND=1"
            )
            if "!ALREADY_FOUND!"=="0" (
                set /a INDEX+=1
                set "PYTHON_PATHS[!INDEX!]=%%D\python.exe"
                set "PYTHON_BASE[!INDEX!]=%%D"
                echo !INDEX!. %%D
            )
        )
    )
)

@REM Check additional locations (Program Files)
for %%P in ("C:\Python*" "C:\Program Files\Python*" "C:\Program Files (x86)\Python*") do (
    if exist %%P (
        for /d %%D in (%%P) do (
            if exist "%%D\python.exe" (
                @REM Check if this path is already in our list
                set "ALREADY_FOUND=0"
                for /L %%N in (1,1,!INDEX!) do (
                    if "!PYTHON_PATHS[%%N]!"=="%%D\python.exe" set "ALREADY_FOUND=1"
                )
                if "!ALREADY_FOUND!"=="0" (
                    set /a INDEX+=1
                    set "PYTHON_PATHS[!INDEX!]=%%D\python.exe"
                    set "PYTHON_BASE[!INDEX!]=%%D"
                    echo !INDEX!. %%D
                )
            )
        )
    )
)

if !INDEX! EQU 0 (
    echo No Python installations found.
    echo Please ensure Python is installed and try running this script as Administrator.
    pause
    exit /b 1
)

echo.
echo Python installations found: !INDEX!

@REM Step 3: Prompt user to select a Python version
:select_python
echo.
set /p CHOICE=Enter the number of the Python version to use for venv (1-!INDEX!): 

@REM Validate user input
if "!CHOICE!"=="" (
    echo Error: No input provided. Please enter a number between 1 and !INDEX!.
    goto :select_python
)

@REM Check if input is numeric
echo !CHOICE!| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo Error: Please enter a valid number between 1 and !INDEX!.
    goto :select_python
)

if !CHOICE! LSS 1 (
    echo Error: Invalid selection. Please enter a number between 1 and !INDEX!.
    goto :select_python
)
if !CHOICE! GTR !INDEX! (
    echo Error: Invalid selection. Please enter a number between 1 and !INDEX!.
    goto :select_python
)

@REM Set selected Python path
set "SELECTED_PYTHON=!PYTHON_PATHS[%CHOICE%]!"
set "SELECTED_BASE=!PYTHON_BASE[%CHOICE%]!"

@REM Verify selected Python exists
if not defined SELECTED_PYTHON (
    echo Error: Python path not found for choice !CHOICE!.
    pause
    exit /b 1
)
if not exist "!SELECTED_PYTHON!" (
    echo Error: Selected Python executable not found at !SELECTED_PYTHON!.
    pause
    exit /b 1
)

echo.
echo Selected Python: !SELECTED_PYTHON!
echo Base directory: !SELECTED_BASE!

@REM Test the selected Python
echo Testing selected Python...
"!SELECTED_PYTHON!" --version
if errorlevel 1 (
    echo Error: Selected Python is not working properly.
    pause
    exit /b 1
)

@REM Step 4: Create a new virtual environment
set VENV_NAME=venv

echo Creating virtual environment "%VENV_NAME%" using !SELECTED_PYTHON!...
"!SELECTED_PYTHON!" -m venv %VENV_NAME%

if not exist "%VENV_NAME%" (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

@REM Step 5: Copy Include and Libs folders to the Venv (Triton)
echo Copying Include and Libs folders from !SELECTED_BASE! to %VENV_NAME%...
xcopy /E /I /Y "!SELECTED_BASE!\Include" "%VENV_NAME%\Include\"
xcopy /E /I /Y "!SELECTED_BASE!\libs" "%VENV_NAME%\libs\"

@REM Copy runtime DLLs if they exist
if exist "!SELECTED_BASE!\vcruntime140.dll" (
    xcopy /I /Y "!SELECTED_BASE!\vcruntime140.dll" "%VENV_NAME%\Scripts\"
)
if exist "!SELECTED_BASE!\vcruntime140_1.dll" (
    xcopy /I /Y "!SELECTED_BASE!\vcruntime140_1.dll" "%VENV_NAME%\Scripts\"
)

echo Virtual environment "%VENV_NAME%" created successfully!
echo Include and libs folders copied.

call venv\Scripts\activate.bat
echo Venv Activated

@REM -------------------------------------------------------------------------------------------------
@REM Installing packages for the Venv and requirements for SageAttention including Pytorch
python -m pip install --upgrade pip

pause

@REM Checking for installed CUDA version and installing latest relevant Pytorch for it
setlocal enabledelayedexpansion

:: Step 1: Get the CUDA version using nvcc --version
for /f "tokens=5 delims= " %%A in ('nvcc --version ^| findstr /C:"release"') do (
    for /f "tokens=1 delims=," %%B in ("%%A") do set cuda_version=%%B
)

REM Extract major version
for /f "tokens=1 delims=." %%a in ("%cuda_version%") do set cuda_major=%%a

REM Extract minor version
for /f "tokens=2 delims=." %%b in ("%cuda_version%") do set cuda_minor=%%b

set cuda_version=!cuda_major!.!cuda_minor!

echo Detected CUDA Version: %cuda_version%

:: Step 3: Determine the closest supported CUDA version (12.4, 12.6, or 12.8)
if "%cuda_version%"=="12.4" (
    set target_cuda_version=12.4
) else if "%cuda_version%"=="12.6" (
    set target_cuda_version=12.6
) else if "%cuda_version%"=="12.8" (
    set target_cuda_version=12.8
) else if "%cuda_version%" gtr "12.6" (
    set target_cuda_version=12.8
) else (
    set target_cuda_version=12.4
)

echo Installing PyTorch for CUDA %target_cuda_version%

:: Step 4: Construct the PyTorch installation command based on the determined CUDA version
if "%target_cuda_version%"=="12.4" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
) else if "%target_cuda_version%"=="12.6" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
) else if "%target_cuda_version%"=="12.8" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
) else (
    echo Unsupported CUDA version: %cuda_version%
    pause
    exit /b 1
)
echo Installation complete!
echo PyTorch with CUDA version %target_cuda_version% has been installed.

@REM Step 5: Detect Python version and install corresponding wheels for insightface, flash-attn, and deepspeed
echo Detecting Python version...
for /f "tokens=2 delims= " %%A in ('python --version') do set python_version=%%A
for /f "tokens=1,2 delims=." %%A in ("%python_version%") do (
    set python_major=%%A
    set python_minor=%%B
)
set python_version=%python_major%.%python_minor%

echo Detected Python version: %python_version%

:: Map Python version to wheel tag (cp310, cp311, cp312)
if "%python_version%"=="3.10" (
    set wheel_tag=cp310
) else if "%python_version%"=="3.11" (
    set wheel_tag=cp311
) else if "%python_version%"=="3.12" (
    set wheel_tag=cp312
) else (
    echo Unsupported Python version: %python_version%. Supported versions are 3.10, 3.11, and 3.12.
    pause
    exit /b 1
)

echo Installing wheels for Python %python_version% (wheel tag: %wheel_tag%)...

:: Install insightface
if "%wheel_tag%"=="cp310" (
    pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/insightface-0.7.3-cp310-cp310-win_amd64.whl
) else if "%wheel_tag%"=="cp311" (
    pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/insightface-0.7.3-cp311-cp311-win_amd64.whl
) else if "%wheel_tag%"=="cp312" (
    pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/insightface-0.7.3-cp312-cp312-win_amd64.whl
)

:: Install flash-attn
if "%wheel_tag%"=="cp310" (
    pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/flash_attn-2.7.4.post1-cp310-cp310-win_amd64.whl
) else if "%wheel_tag%"=="cp311" (
    pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/flash_attn-2.7.4.post1-cp311-cp311-win_amd64.whl
) else if "%wheel_tag%"=="cp312" (
    pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/flash_attn-2.7.4.post1-cp312-cp312-win_amd64.whl
)

:: Install deepspeed
if "%wheel_tag%"=="cp310" (
    pip install https://files.pythonhosted.org/packages/15/b0/be6cc74fd1e23da20d6c34db923858a8ae5017d39a13dedc188a935c646a/deepspeed-0.16.5-cp310-cp310-win_amd64.whl
) else if "%wheel_tag%"=="cp311" (
    pip install https://mirrors.aliyun.com/pypi/packages/0b/78/9a87d137f83f53dd57f425ac174723c746c53029bf6cdce22eb9e805e3f0/deepspeed-0.16.5-cp311-cp311-win_amd64.whl
) else if "%wheel_tag%"=="cp312" (
    pip install https://mirrors.aliyun.com/pypi/packages/80/f8/a9b1fca237fe8ea25f866956778280d0b735350f29fb6f4e0b548aa02a5f/deepspeed-0.16.5-cp312-cp312-win_amd64.whl
)

echo Wheel installation for insightface, flash-attn, and deepspeed complete!

@REM Install the rest of the requirements for the Venv, Triton, and SageAttention
pip install --upgrade pip
pip install --upgrade huggingface-hub
pip install -r requirements.txt
pip install onnxruntime-gpu
pip install wheel
pip install setuptools
pip install packaging
pip install ninja
pip install "accelerate>=1.1.1"
pip install "diffusers>=0.31.0"
pip install "transformers>=4.39.3"
pip install --upgrade diffusers huggingface-hub
pip install mediapipe>=0.10.8
pip install transformers
pip install huggingface-hub
pip install omegaconf
pip install einops
pip install opencv-python
pip install face-alignment
pip install decord
pip install ffmpeg-python>=0.2.0
pip install safetensors
pip install soundfile
pip install pytorch-lightning
pip install pyaudio sounddevice

python -m ensurepip --upgrade
python -m pip install --upgrade setuptools

echo Pytorch installed and requirements for the venv, Triton, and SageAttention installed.pause

pause
@REM --------------------------------------------------------------------------------------------------
@REM Install Triton Wheel for Triton & install

setlocal enabledelayedexpansion

@REM Step 1: Determine Python Major and Minor Version and display it to User
for /f "tokens=2 delims= " %%i in ('python --version') do (
    set py_version=%%i
)
for /f "tokens=1,2 delims=." %%a in ("!py_version!") do (
    set py_major_version=%%a
    set py_minor_version=%%b
)

@REM Step 2: Display the Installed Python Version (without minor version)
echo Detected Python Version: !py_major_version!.!py_minor_version!
echo !py_version!

@REM Step 3: Detect PyTorch version using Python
for /f "delims=" %%A in ('python -c "import torch; print(torch.__version__)" 2^>nul') do set "PYTORCH_VERSION=%%A"

@REM Step 4: Extract major and minor version (e.g., 2.5 from 2.5.1)
for /f "tokens=1,2 delims=." %%B in ("%PYTORCH_VERSION%") do (
    set "PYTORCH_MAJOR=%%B"
    set "PYTORCH_MINOR=%%C"
    set "PYTORCH_VERSION_SHORT=%%B.%%C"
)

@REM Step 5: Check if PyTorch is installed
if not defined PYTORCH_VERSION (
    echo ERROR: PyTorch is not installed. Please install PyTorch first.
    pause
    exit /b
)

echo Detected PyTorch version: %PYTORCH_VERSION_SHORT%

@REM Step 6: Restrict Triton versions based on PyTorch version
if "%PYTORCH_MAJOR%"=="2" (
    if "%PYTORCH_MINOR%" GEQ "6" (
        echo PyTorch 2.6+ detected. All Triton versions available.
        set "OPTION1=1 - Triton 3.2.0"
        set "OPTION2=2 - Triton 3.1.0"
    ) else if "%PYTORCH_MINOR%" GEQ "4" (
        echo PyTorch 2.4 or 2.5 detected. Triton 3.2.0 is not supported.
        set "OPTION1="
        set "OPTION2=2 - Triton 3.1.0"
    ) else (
        echo WARNING: PyTorch %PYTORCH_VERSION_SHORT% is too old! Triton 3.1.0 and 3.2.0 require PyTorch 2.4+.
        echo Only Triton 3.0.0 is available, unsure if it will work.
        set "OPTION1="
        set "OPTION2="
    )
) else (
    echo ERROR: PyTorch version not supported. Only PyTorch 2.x is allowed.
    pause
    exit /b
)

set "OPTION3=3 - Triton 3.0.0"

@REM Step 7: Display options
echo Select the package you want to download: 
if defined OPTION1 echo %OPTION1%
if defined OPTION2 echo %OPTION2%
if defined OPTION3 echo %OPTION3%

@REM Step 8: Get user choice
set /p CHOICE=Enter your choice: 

@REM Step 9: Validate selection
if "%CHOICE%"=="1" if not defined OPTION1 echo Invalid choice. Exiting. & exit /b
if "%CHOICE%"=="2" if not defined OPTION2 echo Invalid choice. Exiting. & exit /b
if "%CHOICE%"=="3" if not defined OPTION3 echo Invalid choice. Exiting. & exit /b

echo You selected Triton %CHOICE%.
if "%CHOICE%"=="1" echo Note: Triton 3.2.0 works best with PyTorch 2.6+. Upgrade recommended!

@REM Step 10: Map Python versions to corresponding wheel URLs
set WHEEL_URL=

if "%CHOICE%"=="1" (
    if "!py_major_version!.!py_minor_version!"=="3.10" set WHEEL_URL=https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp310-cp310-win_amd64.whl
    if "!py_major_version!.!py_minor_version!"=="3.12" set WHEEL_URL=https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp312-cp312-win_amd64.whl
    if "!py_major_version!.!py_minor_version!"=="3.13" set WHEEL_URL=https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp313-cp313-win_amd64.whl
)

if "%CHOICE%"=="2" (
    if "!py_major_version!.!py_minor_version!"=="3.10" set WHEEL_URL=https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post9/triton-3.1.0-cp310-cp310-win_amd64.whl
    if "!py_major_version!.!py_minor_version!"=="3.11" set WHEEL_URL=https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post9/triton-3.1.0-cp311-cp311-win_amd64.whl
    if "!py_major_version!.!py_minor_version!"=="3.12" set WHEEL_URL=https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post9/triton-3.1.0-cp312-cp312-win_amd64.whl
)

if "%CHOICE%"=="3" (
    if "!py_major_version!.!py_minor_version!"=="3.10" set WHEEL_URL=https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post9/triton-3.0.0-cp310-cp310-win_amd64.whl
    if "!py_major_version!.!py_minor_version!"=="3.11" set WHEEL_URL=https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post9/triton-3.0.0-cp311-cp311-win_amd64.whl
    if "!py_major_version!.!py_minor_version!"=="3.12" set WHEEL_URL=https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post9/triton-3.0.0-cp312-cp312-win_amd64.whl
)

@REM Step 11: Validate and download the selected wheel
if "%WHEEL_URL%"=="" (
    echo No compatible wheel found for Python %PYTHON_MAJOR%.%PYTHON_MINOR% or invalid choice.
    exit /b
)

echo Installing Triton package for Python %PYTHON_MAJOR%.%PYTHON_MINOR%...
pip install %WHEEL_URL%

@REM Step 12: Deleting Tritons cached files as these can make it fault

setlocal

set "TRITON_CACHE=C:\Users\%USERNAME%\.triton\cache"
set "TORCHINDUCTOR_CACHE=C:\Users\%USERNAME%\AppData\Local\Temp\torchinductor_%USERNAME%\triton"

if exist "%TRITON_CACHE%" (
    echo Deleting .triton cache...
    rmdir /s /q "%TRITON_CACHE%" 2>nul
    mkdir "%TRITON_CACHE%"
    echo .Triton cache cleared.
) else (
    echo .Triton cache folder not found.
)

if exist "%TORCHINDUCTOR_CACHE%" (
    echo Deleting torchinductor cache...
    rmdir /s /q "%TORCHINDUCTOR_CACHE%" 2>nul
    mkdir "%TORCHINDUCTOR_CACHE%"
    echo Torchinductor cache cleared.
) else (
    echo Torchinductor cache folder not found.
)

echo Triton installed and caches cleared

pause

@REM --------------------------------------------------------------------------------------------------
@REM Install SageAttention

cd venv
git clone https://github.com/thu-ml/SageAttention
cd SageAttention
set MAX_JOBS=4

echo Make a coffee and ignore various error comments whilst SageAttention compiles

python.exe setup.py install


@REM Cleaning up SageAttention
cd ..
rmdir /s /q SageAttention
echo SageAttention installed and cleared up

pause

@REM --------------------------------------------------------------------------------------------------
@REM Deactivate venv
@REM Deactivate

@REM --------------------------------------------------------------------------------------------------
@REM Make a start bat file for Comfy
setlocal

cd ..
cd ..

@REM Step 1: Define the path for the new batch file
set "new_batch_file=Run_Comfyui.bat"

@REM Step 2: Create the new Comfy startup batch file
(
echo @echo off
echo cd ComfyUI
echo call venv\Scripts\activate.bat
echo echo Venv Activated
echo .\venv\Scripts\python.exe -s main.py --fast --windows-standalone-build
echo pause
) > "%new_batch_file%"

@REM Step 3: Check if the new batch file was created successfully
if exist "%new_batch_file%" (
    echo The file %new_batch_file% has been created successfully.
) else (
    echo Failed to create the file %new_batch_file%.
)

@REM --------------------------------------------------------------------------------------------------
@REM Create a batch file to auto open a CMD window and activate the Venv
@REM Step 1: Define the path for the new batch file
set "new_batch_file2=Activate_Venv.bat"

@REM Step 2: Create the new batch file with the specified content
(
echo @echo off
echo cd ComfyUI\venv
echo call .\Scripts\activate.bat
echo echo Venv Activated
echo cmd.exe /k
) > "%new_batch_file2%"

@REM Step 3: Check if the new batch file was created successfully
if exist "%new_batch_file2%" (
    echo The file %new_batch_file2% has been created successfully.
) else (
    echo Failed to create the file %new_batch_file2%.
)

@REM --------------------------------------------------------------------------------------------------
@REM Create a batch file to update Comfy via git pull
@REM Step 1: Define the path for the new batch file
set "new_batch_file3=Update_Comfy.bat"

@REM Step 2: Create the new batch file with the specified content
(
echo @echo off
echo cd ComfyUI
echo git pull
echo pause
) > "%new_batch_file3%"

@REM Step 3: Check if the new batch file was created successfully
if exist "%new_batch_file3%" (
    echo The file %new_batch_file3% has been created successfully.
) else (
    echo Failed to create the file %new_batch_file3%.
)

echo Three bat files saved to install folder 
echo   1. ComfyUI start 
echo   2. Activate the venv for manual input 
echo   3. Update via git pull
pause

@REM --------------------------------------------------------------------------------------------------
@REM Installing Comfy Manager rather than faffing around 

cd ComfyUI\custom_nodes

git clone https://github.com/ltdrdata/ComfyUI-Manager.git
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-Manager. Continuing...
)

@REM Installing additional custom nodes for ComfyUI
echo Installing additional custom nodes for ComfyUI...

@REM Clone each custom node and install requirements.txt if it exists
echo Cloning comfyui_controlnet_aux...
git clone https://github.com/Fannovel16/comfyui_controlnet_aux
if errorlevel 1 (
    echo Warning: Failed to clone comfyui_controlnet_aux. Continuing...
) else (
    cd comfyui_controlnet_aux
    if exist "requirements.txt" (
        echo Installing requirements for comfyui_controlnet_aux...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for comfyui_controlnet_aux. Continuing...
        )
    ) else (
        echo No requirements.txt found for comfyui_controlnet_aux.
    )
    cd ..
)

echo Cloning ComfyUI-Custom-Scripts...
git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-Custom-Scripts. Continuing...
) else (
    cd ComfyUI-Custom-Scripts
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-Custom-Scripts...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-Custom-Scripts. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-Custom-Scripts.
    )
    cd ..
)

echo Cloning ComfyUI-Impact-Pack...
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-Impact-Pack. Continuing...
) else (
    cd ComfyUI-Impact-Pack
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-Impact-Pack...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-Impact-Pack. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-Impact-Pack.
    )
    cd ..
)

echo Cloning rgthree-comfy...
git clone https://github.com/rgthree/rgthree-comfy
if errorlevel 1 (
    echo Warning: Failed to clone rgthree-comfy. Continuing...
) else (
    cd rgthree-comfy
    if exist "requirements.txt" (
        echo Installing requirements for rgthree-comfy...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for rgthree-comfy. Continuing...
        )
    ) else (
        echo No requirements.txt found for rgthree-comfy.
    )
    cd ..
)

echo Cloning ComfyUI-KJNodes...
git clone https://github.com/kijai/ComfyUI-KJNodes
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-KJNodes. Continuing...
) else (
    cd ComfyUI-KJNodes
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-KJNodes...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-KJNodes. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-KJNodes.
    )
    cd ..
)

echo Cloning ComfyUI-Florence2...
git clone https://github.com/kijai/ComfyUI-Florence2
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-Florence2. Continuing...
) else (
    cd ComfyUI-Florence2
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-Florence2...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-Florence2. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-Florence2.
    )
    cd ..
)

echo Cloning ComfyUI_essentials...
git clone https://github.com/cubiq/ComfyUI_essentials
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI_essentials. Continuing...
) else (
    cd ComfyUI_essentials
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI_essentials...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI_essentials. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI_essentials.
    )
    cd ..
)

echo Cloning ComfyUI-Inspire-Pack...
git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-Inspire-Pack. Continuing...
) else (
    cd ComfyUI-Inspire-Pack
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-Inspire-Pack...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-Inspire-Pack. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-Inspire-Pack.
    )
    cd ..
)

echo Cloning ComfyUI-Impact-Subpack...
git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-Impact-Subpack. Continuing...
) else (
    cd ComfyUI-Impact-Subpack
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-Impact-Subpack...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-Impact-Subpack. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-Impact-Subpack.
    )
    cd ..
)

echo Cloning comfyui-various...
git clone https://github.com/jamesWalker55/comfyui-various
if errorlevel 1 (
    echo Warning: Failed to clone comfyui-various. Continuing...
) else (
    cd comfyui-various
    if exist "requirements.txt" (
        echo Installing requirements for comfyui-various...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for comfyui-various. Continuing...
        )
    ) else (
        echo No requirements.txt found for comfyui-various.
    )
    cd ..
)

echo Cloning comfyui-tensorops...
git clone https://github.com/un-seen/comfyui-tensorops
if errorlevel 1 (
    echo Warning: Failed to clone comfyui-tensorops. Continuing...
) else (
    cd comfyui-tensorops
    if exist "requirements.txt" (
        echo Installing requirements for comfyui-tensorops...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for comfyui-tensorops. Continuing...
        )
    ) else (
        echo No requirements.txt found for comfyui-tensorops.
    )
    cd ..
)

echo Cloning ComfyUI-GGUF...
git clone https://github.com/city96/ComfyUI-GGUF
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-GGUF. Continuing...
) else (
    cd ComfyUI-GGUF
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-GGUF...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-GGUF. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-GGUF.
    )
    cd ..
)

echo Cloning ComfyUI-AdvancedLivePortrait...
git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-AdvancedLivePortrait. Continuing...
) else (
    cd ComfyUI-AdvancedLivePortrait
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-AdvancedLivePortrait...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-AdvancedLivePortrait. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-AdvancedLivePortrait.
    )
    cd ..
)

echo Cloning ComfyUI_FaceAnalysis...
git clone https://github.com/cubiq/ComfyUI_FaceAnalysis
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI_FaceAnalysis. Continuing...
) else (
    cd ComfyUI_FaceAnalysis
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI_FaceAnalysis...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI_FaceAnalysis. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI_FaceAnalysis.
    )
    cd ..
)

echo Cloning masquerade-nodes-comfyui...
git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui
if errorlevel 1 (
    echo Warning: Failed to clone masquerade-nodes-comfyui. Continuing...
) else (
    cd masquerade-nodes-comfyui
    if exist "requirements.txt" (
        echo Installing requirements for masquerade-nodes-comfyui...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for masquerade-nodes-comfyui. Continuing...
        )
    ) else (
        echo No requirements.txt found for masquerade-nodes-comfyui.
    )
    cd ..
)

echo Cloning comfyui_face_parsing...
git clone https://github.com/Ryuukeisyou/comfyui_face_parsing
if errorlevel 1 (
    echo Warning: Failed to clone comfyui_face_parsing. Continuing...
) else (
    cd comfyui_face_parsing
    if exist "requirements.txt" (
        echo Installing requirements for comfyui_face_parsing...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for comfyui_face_parsing. Continuing...
        )
    ) else (
        echo No requirements.txt found for comfyui_face_parsing.
    )
    cd ..
)

echo Cloning ComfyUI_tinyterraNodes...
git clone https://github.com/TinyTerra/ComfyUI_tinyterraNodes
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI_tinyterraNodes. Continuing...
) else (
    cd ComfyUI_tinyterraNodes
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI_tinyterraNodes...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI_tinyterraNodes. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI_tinyterraNodes.
    )
    cd ..
)

echo Cloning Save_Florence2_Bulk_Prompts...
git clone https://github.com/Pixelailabs/Save_Florence2_Bulk_Prompts
if errorlevel 1 (
    echo Warning: Failed to clone Save_Florence2_Bulk_Prompts. Continuing...
) else (
    cd Save_Florence2_Bulk_Prompts
    if exist "requirements.txt" (
        echo Installing requirements for Save_Florence2_Bulk_Prompts...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for Save_Florence2_Bulk_Prompts. Continuing...
        )
    ) else (
        echo No requirements.txt found for Save_Florence2_Bulk_Prompts.
    )
    cd ..
)

echo Cloning ComfyUI_LayerStyle_Advance...
git clone https://github.com/chflame163/ComfyUI_LayerStyle_Advance
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI_LayerStyle_Advance. Continuing...
) else (
    cd ComfyUI_LayerStyle_Advance
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI_LayerStyle_Advance...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI_LayerStyle_Advance. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI_LayerStyle_Advance.
    )
    cd ..
)

echo Cloning ComfyUI_LayerStyle...
git clone https://github.com/chflame163/ComfyUI_LayerStyle
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI_LayerStyle. Continuing...
) else (
    cd ComfyUI_LayerStyle
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI_LayerStyle...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI_LayerStyle. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI_LayerStyle.
    )
    cd ..
)

echo Cloning ComfyUI-Easy-Use...
git clone https://github.com/yolain/ComfyUI-Easy-Use
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-Easy-Use. Continuing...
) else (
    cd ComfyUI-Easy-Use
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-Easy-Use...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-Easy-Use. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-Easy-Use.
    )
    cd ..
)

echo Cloning ComfyUI-VideoHelperSuite...
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-VideoHelperSuite. Continuing...
) else (
    cd ComfyUI-VideoHelperSuite
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-VideoHelperSuite...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-VideoHelperSuite. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-VideoHelperSuite.
    )
    cd ..
)

echo Cloning ComfyUI-Frame-Interpolation...
git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-Frame-Interpolation. Continuing...
) else (
    cd ComfyUI-Frame-Interpolation
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-Frame-Interpolation...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-Frame-Interpolation. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-Frame-Interpolation.
    )
    cd ..
)

echo Cloning ComfyUI-wanBlockswap...
git clone https://github.com/orssorbit/ComfyUI-wanBlockswap
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-wanBlockswap. Continuing...
) else (
    cd ComfyUI-wanBlockswap
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-wanBlockswap...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-wanBlockswap. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-wanBlockswap.
    )
    cd ..
)

echo Cloning ComfyUI-SparkTTS...
git clone https://github.com/1038lab/ComfyUI-SparkTTS
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-SparkTTS. Continuing...
) else (
    cd ComfyUI-SparkTTS
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-SparkTTS...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-SparkTTS. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-SparkTTS.
    )
    cd ..
)

echo Cloning ComfyUI_SLK_joy_caption_two...
git clone https://github.com/EvilBT/ComfyUI_SLK_joy_caption_two
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI_SLK_joy_caption_two. Continuing...
) else (
    cd ComfyUI_SLK_joy_caption_two
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI_SLK_joy_caption_two...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI_SLK_joy_caption_two. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI_SLK_joy_caption_two.
    )
    cd ..
)

echo Cloning ComfyUI-WanVideoWrapper...
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper
if errorlevel 1 (
    echo Warning: Failed to clone ComfyUI-WanVideoWrapper. Continuing...
) else (
    cd ComfyUI-WanVideoWrapper
    if exist "requirements.txt" (
        echo Installing requirements for ComfyUI-WanVideoWrapper...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for ComfyUI-WanVideoWrapper. Continuing...
        )
    ) else (
        echo No requirements.txt found for ComfyUI-WanVideoWrapper.
    )
    cd ..
)

echo Cloning audio-separation-nodes-comfyui...
git clone https://github.com/christian-byrne/audio-separation-nodes-comfyui
if errorlevel 1 (
    echo Warning: Failed to clone audio-separation-nodes-comfyui. Continuing...
) else (
    cd audio-separation-nodes-comfyui
    if exist "requirements.txt" (
        echo Installing requirements for audio-separation-nodes-comfyui...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo Warning: Failed to install requirements for audio-separation-nodes-comfyui. Continuing...
        )
    ) else (
        echo No requirements.txt found for audio-separation-nodes-comfyui.
    )
    cd ..
)


echo Custom Nodes Cloning completed
echo Script Made by Aiconomist - Please run Models_Downloader.bat before starting ComfyUI

pause
cmd.exe /k