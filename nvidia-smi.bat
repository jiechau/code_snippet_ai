@echo off

for /l %%x in (1, 1, 60) do (
    cls
    REM echo %%x
    nvidia-smi.exe
    REM timeout 1
    ping -n 3 localhost >nul
)
