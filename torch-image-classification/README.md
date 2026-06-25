# PyTorch — Image Classification

Image classification (MobileNet-style) in PyTorch, with GPU and CPU variants and
inference-only scripts. Also includes a FLOPs/params measurement script.

## Setup

Use a `pt39` (or `pt311`) virtual environment — works on GPU or CPU. See
[../venv-setup/](../venv-setup/).

```bash
source pt39/bin/activate
pip install -r requirements.pt39.pip.txt
```

## Run

```bash
python pt_mn.py        # train + classify (GPU)
python pt_mni.py       # inference only (GPU)
python pt_mn_cpu.py    # train + classify (CPU)
python pt_mni_cpu.py   # inference only (CPU)
python pt_mn_flops.py  # measure model FLOPs / params
```

> For the `*i` (inference-only) scripts you need to unmark the last line of the
> matching training script (`pt_mn.py` / `pt_mn_cpu.py`) to save the model first.

## Files

- [pt_mn.py](pt_mn.py) / [pt_mni.py](pt_mni.py) — GPU train / inference.
- [pt_mn_cpu.py](pt_mn_cpu.py) / [pt_mni_cpu.py](pt_mni_cpu.py) — CPU train / inference.
- [pt_mn_flops.py](pt_mn_flops.py) — FLOPs / params measurement.
- `requirements.pt39.pip.txt` / `requirements.pt311.pip.txt` — Python 3.9 / 3.11 dependencies.
