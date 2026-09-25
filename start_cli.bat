@echo off
title PDF & Web Analyzer - Terminal CLI Mode
cls
echo ======================================================================
echo          PDF & WEB ANALYZER - TERMINAL CLI MODE
echo ======================================================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python Main.py

pause
