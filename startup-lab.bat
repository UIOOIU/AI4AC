@echo off

REM 设置项目目录路径
set PROJECT_DIR=%~dp0

REM 切换到项目目录
cd %PROJECT_DIR%

@echo off
REM 检查进程是否存在，存在则kill
REM taskkill /f /im jupyter-notebook.exe
tasklist | find "jupyter-lab.exe" > nul && taskkill /f /im jupyter-lab.exe


REM 启动Jupyter Notebook
cmd.exe /k "conda activate && jupyter lab"
