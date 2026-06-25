#!/bin/bash

python --version
nvcc --version
nvidia-smi
nvidia-smi -q
lspci
lshw -C display
cat /proc/meminfo
cat /proc/cpuinfo
