# Hugging Face — TFAutoModelForImageClassification

Image classification with Hugging Face Transformers using the TensorFlow
`TFAutoModelForImageClassification` API, with a separate inference script.

## Setup

Use the `hf39tf` virtual environment (TensorFlow backend):

```bash
/usr/local/bin/python3.9 -m venv --system-site-packages hf39tf
source hf39tf/bin/activate
pip install --upgrade pip
pip install -r requirements.hf39tf.pip.txt
```

## Run

```bash
python hf_TFAutoModelForImageClassification.py            # train
python hf_TFAutoModelForImageClassification_inference.py  # inference
```

## Files

- [hf_TFAutoModelForImageClassification.py](hf_TFAutoModelForImageClassification.py) — train.
- [hf_TFAutoModelForImageClassification_inference.py](hf_TFAutoModelForImageClassification_inference.py) — inference.
- `requirements.hf39tf.pip.txt` — dependencies.
