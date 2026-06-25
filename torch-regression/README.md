# PyTorch — Regression

A simple deep neural network regression example in PyTorch.

## Setup

Use a `pt39` (or `pt311`) virtual environment — works on GPU or CPU. See
[../venv-setup/](../venv-setup/) for the general pattern.

```bash
/usr/local/bin/python3.9 -m venv --system-site-packages pt39
source pt39/bin/activate
pip install -r requirements.pt39.pip.txt
```

## Run

```bash
python pt_dnn.py
```

## Files

- [pt_dnn.py](pt_dnn.py) — DNN regression training/inference.
- `requirements.pt39.pip.txt` / `requirements.pt311.pip.txt` — Python 3.9 / 3.11 dependencies.
