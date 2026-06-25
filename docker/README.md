# Docker — TensorFlow / PyTorch GPU Containers

Notes for running the snippets in this repo inside official TensorFlow / PyTorch
Docker images, mainly to sanity-check GPU throughput.

## TensorFlow

```bash
# GPU
docker run -it --rm --gpus all tensorflow/tensorflow:latest-gpu bash
# CPU  (Ubuntu 22.04.3 LTS "Jammy Jellyfish")
docker run -it --rm tensorflow/tensorflow:latest bash

# inside the container:
nvidia-smi
apt update; apt install git
git clone https://gitlab.com/jiechau/code_snippet_ai.git
cd code_snippet_ai
python tf_mn.py        # every epoch should be < 5 sec
```

## PyTorch

```bash
# GPU or CPU
docker run -it --rm --gpus all nvcr.io/nvidia/pytorch:23.10-py3 bash

# inside the container:
git clone https://gitlab.com/jiechau/code_snippet_ai.git
nvidia-smi
cd code_snippet_ai
python pt_mn.py        # every epoch should be < 10 sec
python pt_mn_cpu.py
```
