@echo off

REM 设置项目目录路径
set PROJECT_DIR=%~dp0

REM 切换到项目目录
cd %PROJECT_DIR%

REM 激活Python环境（假设使用Anaconda）
call D:\DevSoft\python-venv\research-venv\Scripts\activate.bat

@echo off
REM 检查进程是否存在
REM tasklist | find "jupyter-notebook.exe" > nul
REM if not errorlevel 1 (
REM	taskkill /f /im jupyter-notebook.exe
REM )

taskkill /f /im jupyter-notebook.exe

REM 启动Jupyter Notebook
jupyter notebook
