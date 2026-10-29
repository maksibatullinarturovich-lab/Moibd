@echo off
chcp 65001 > nul
cd /d "%~dp0\app"
streamlit run app.py
pause
