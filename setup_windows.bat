@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" tools\verify_package.py
exit /b %errorlevel%
