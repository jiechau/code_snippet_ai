# Environment Check

Small helper scripts to verify a machine's Python / TensorFlow / PyTorch / CUDA setup
and watch the GPU.

## Files

- [pt_check.py](pt_check.py) — print Python, PyTorch, torchvision versions and CUDA availability; run a tensor on the GPU.
- [tf_check.py](tf_check.py) — print Python, TensorFlow, Keras versions and list local GPU devices.
- [tf_check.sh](tf_check.sh) — Linux system info: `python`, `nvcc`, `nvidia-smi`, `lspci`, `lshw`, CPU/mem info.
- [tf_check.bat](tf_check.bat) — Windows: Python version and key `pip freeze` packages (tensorflow / numpy / proto).
- [nvidia-smi.bat](nvidia-smi.bat) — Windows: loop `nvidia-smi` to watch the GPU (poor man's `watch`).

## Run

```bash
python pt_check.py
python tf_check.py
bash tf_check.sh        # Linux
```
