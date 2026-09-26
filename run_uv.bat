@echo off
setlocal
title AI Video Dubbing Pro (UV Mode)

:: Add uv directory to PATH
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
set "PYTHONUNBUFFERED=1"

echo ========================================================
echo   Starting AI Video Dubbing Pro (UV Mode)
echo ========================================================
echo.

:: 1. Check if uv is in system PATH
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] uv found in PATH.
    uv pip install --quiet yt-dlp
    echo [RUN] uv run --no-sync python gui.py
    echo.
    uv run --no-sync python gui.py
    goto :handle_exit
)

:: 2. Check standard uv install locations
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    echo [OK] uv found at %USERPROFILE%\.local\bin\uv.exe
    "%USERPROFILE%\.local\bin\uv.exe" pip install --quiet yt-dlp
    echo [RUN] "%USERPROFILE%\.local\bin\uv.exe" run --no-sync python gui.py
    echo.
    "%USERPROFILE%\.local\bin\uv.exe" run --no-sync python gui.py
    goto :handle_exit
)

if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    echo [OK] uv found at %USERPROFILE%\.cargo\bin\uv.exe
    "%USERPROFILE%\.cargo\bin\uv.exe" pip install --quiet yt-dlp
    echo [RUN] "%USERPROFILE%\.cargo\bin\uv.exe" run --no-sync python gui.py
    echo.
    "%USERPROFILE%\.cargo\bin\uv.exe" run --no-sync python gui.py
    goto :handle_exit
)

:: 3. Fallback to .venv if uv not found
echo [WARN] uv was not found.
echo [RUN] .venv\Scripts\python.exe gui.py
echo.
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe gui.py
) else (
    echo [ERROR] Neither uv nor .venv found!
    echo To install uv, run in PowerShell:
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

:handle_exit
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%.
    pause
)
endlocal
