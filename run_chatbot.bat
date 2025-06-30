@echo off
cd /d C:\Users\ASUS\Desktop\ppu-chatbot

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Ensure gradio is installed
pip install gradio --quiet

:: Run chatbot in new window
start "PPU Chatbot" cmd /k python web_chatbot.py
