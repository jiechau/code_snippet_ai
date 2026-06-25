# TensorFlow — Image Classification

Image classification (MobileNet-style) in TensorFlow/Keras, with a separate
inference-only script.

## Setup

Use a `tf39` (GPU) or `tf39cpu` (CPU) virtual environment — see [../venv-setup/](../venv-setup/).

```bash
# GPU
source tf39/bin/activate
pip install -r requirements.tf39.pip.txt
# CPU
source tf39cpu/bin/activate
pip install -r requirements.tf39cpu.pip.txt
```

## Run

```bash
python tf_mn.py    # train + classify
python tf_mni.py   # inference only
```

## Files

- [tf_mn.py](tf_mn.py) — train / classify.
- [tf_mni.py](tf_mni.py) — inference only.
- `requirements.tf39.pip.txt` / `requirements.tf39cpu.pip.txt` — GPU / CPU dependencies.
