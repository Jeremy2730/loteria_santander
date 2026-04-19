@echo off

cd /d %~dp0\..

echo Iniciando predictor...

env\Scripts\python.exe src\sistema_loteria.py

pause