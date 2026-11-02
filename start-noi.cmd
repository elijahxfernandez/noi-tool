@echo off
cd /d "%~dp0"
uv run python -m noi_tool.dashboard --collect
pause
