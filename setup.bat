@echo off
setlocal

echo [SETUP] Checking for uv installation...
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [SETUP] 'uv' not found. Installing via official script...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install uv. Please install it manually: https://docs.astral.sh/uv/
        pause
        exit /b %ERRORLEVEL%
    )

    :: Temporarily add uv to the current session's PATH so the next command works
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    echo [SETUP] uv installed successfully.
) else (
    echo [SETUP] uv is already installed.
)

echo [SETUP] Synchronizing environment with uv...
pushd "%~dp0"
uv sync
if %ERRORLEVEL% neq 0 (
    echo [ERROR] uv sync failed.
    popd
    pause
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Environment is ready.
popd
pause
