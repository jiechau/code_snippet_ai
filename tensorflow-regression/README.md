# TensorFlow — Regression

A simple deep neural network regression example in TensorFlow/Keras, plus a
TensorFlow.js export of the trained model that runs in the browser.

## Setup

Use a `tf39` (GPU) or `tf39cpu` (CPU) virtual environment — see [../venv-setup/](../venv-setup/)
for the general pattern.

```bash
# GPU
/usr/local/bin/python3.9 -m venv --system-site-packages tf39
source tf39/bin/activate
pip install -r requirements.tf39.pip.txt

# CPU
/usr/local/bin/python3.9 -m venv --system-site-packages tf39cpu
source tf39cpu/bin/activate
pip install -r requirements.tf39cpu.pip.txt
```

## Run

```bash
python tf_dnn.py
```

## Files

- [tf_dnn.py](tf_dnn.py) — DNN regression training/inference.
- [tf_dnn_js/](tf_dnn_js/) — TensorFlow.js export (`model.json` + weights + `index.html`) to run the model in a browser.
- `requirements.tf39.pip.txt` / `requirements.tf39cpu.pip.txt` — GPU / CPU dependencies.
