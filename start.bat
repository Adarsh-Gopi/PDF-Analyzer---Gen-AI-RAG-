@echo off
title PDF & Web Analyzer - AI Intelligence
cls
echo ======================================================================
echo          PDF & WEB ANALYZER - AI DOCUMENT & WEB INTELLIGENCE
echo ======================================================================
echo.
echo  [1/2] Checking environment...

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    echo  [OK] Found virtual environment in .venv
    call .venv\Scripts\activate.bat
) else (
    echo  [!] No .venv folder found, using system Python.
)

echo.
echo  [2/2] Launching Streamlit Web Application...
echo.
echo  Access URL: http://localhost:8501
echo  Press Ctrl+C in this window to stop the server.
echo ======================================================================
echo.

streamlit run app.py --server.port 8501

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application failed to launch with standard streamlit command.
    echo Attempting direct python execution...
    python -m streamlit run app.py
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [FATAL] Could not start Streamlit. Please ensure dependencies are installed.
        pause
    )
)
