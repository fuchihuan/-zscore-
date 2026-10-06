@echo off
title Stock-Futures Pair Trading System
cd /d "%~dp0"
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo [ERROR] Python failed. Trying 'py' command...
    py -m streamlit run app.py
)
pause
