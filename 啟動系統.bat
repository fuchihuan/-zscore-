@echo off
title Stock-Futures Pair Trading System
cd /d "%~dp0"
py -3.14 -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.14 failed. Trying default python command...
    python -m streamlit run app.py
)
pause
