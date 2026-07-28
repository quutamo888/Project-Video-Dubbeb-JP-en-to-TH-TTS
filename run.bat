@echo off
echo Starting AI Video Dubbing Pro...
.venv\Scripts\python.exe gui.py
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%.
    pause
)
