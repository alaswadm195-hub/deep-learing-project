@echo off
title Age & Gender AI System - Server
color 0A

echo ======================================================
echo      LAUNCHING AI SYSTEM... PLEASE WAIT...
echo ======================================================

:: 1. فتح الموقع في المتصفح
start "" "index.html"

:: 2. تشغيل السيرفر
echo Starting Python Server...
call venv\Scripts\activate
python app.py

pause