@echo off
title Stock-Futures Pair Trading System
cd /d "%~dp0"
cd "-zscore--main"
if errorlevel 1 (
    echo [ERROR] Cannot find -zscore--main folder!
    pause
    exit /b 1
)
python start.py
if errorlevel 1 (
    echo.
    echo [ERROR] Python failed. Trying 'py' command...
    py start.py
)
pause
