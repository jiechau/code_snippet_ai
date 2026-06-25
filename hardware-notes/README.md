# Hardware & Driver Notes

Assorted notes on the machines and GPU drivers used to run these snippets.

## NVIDIA driver on Ubuntu 22.04

Guide: <https://ivonblog.com/posts/ubuntu-install-nvidia-drivers/>

```bash
apt install nvidia-driver-545
```

GPUs used:
- GeForce GTX 1060 3GB (2016) — `NVIDIA Corporation GP106 [GeForce GTX 1060 3GB] (rev a1)`
- GeForce RTX 3060 12G (2021) — `NVIDIA Corporation GA106 [GeForce RTX 3060 Lite Hash Rate] (rev a1)`

## NVIDIA on ROG Flow X16 (2022) — RTX 3060

Guide: <https://medium.com/@abhig0303/setting-up-tensorflow-with-cuda-for-gpu-on-windows-11-a157db4dae3e>

**ROG Flow X16 (2022) GV601 GV601RM-0042E6900HS**
- 1TB PCIe 4.0 NVMe M.2 SSD
- 8GB DDR5-4800 SO-DIMM x 2
- AMD Ryzen 9 6900HS (8-core / 16-thread, 16MB cache, up to 4.9 GHz)
- NVIDIA GeForce RTX 3060 Laptop GPU, 6GB GDDR6 (ROG Boost up to 1475MHz @ 125W)
- OS: Windows 11 Home

Working TensorFlow-on-GPU combo for this machine:

| Component      | Version            |
| -------------- | ------------------ |
| Python         | 3.8 (3.9 ok too)   |
| TensorFlow     | 2.5 (only 2.5 works) |
| Keras          | 2.5                |
| CUDA Toolkit   | 11.8.0             |
| cuDNN          | 8.6.0              |

```bash
conda create -n py39tf25 python=3.9; conda activate py39tf25
# only pip install works (not conda install)
pip install tensorflow==2.5
```

## Windows conda helpers

`go_conda.bat`:
```bat
%windir%\System32\cmd.exe "/K" C:\ProgramData\anaconda3\Scripts\activate.bat C:\ProgramData\anaconda3
```

`cdj.bat`:
```bat
cd C:\share\jiechau\ml_codes
```

Environments:
```bash
conda env list
conda activate py39tf25
conda activate py39tf25cpu
conda activate py39pt210pip
conda activate py39pt210conda
conda activate py39pt210cpu
conda activate ttt
```
