# Hugging Face — AutoModelForImageClassification

Image classification with Hugging Face Transformers using the PyTorch
`AutoModelForImageClassification` API, with a separate inference script.

## Setup

Use the `hf39pt` virtual environment (PyTorch backend):

```bash
/usr/local/bin/python3.9 -m venv --system-site-packages hf39pt
source hf39pt/bin/activate
pip install --upgrade pip
pip install -r requirements.hf39pt.pip.txt
```

## Run

```bash
python hf_AutoModelForImageClassification.py            # train
python hf_AutoModelForImageClassification_inference.py  # inference
```

## Files

- [hf_AutoModelForImageClassification.py](hf_AutoModelForImageClassification.py) — train.
- [hf_AutoModelForImageClassification_inference.py](hf_AutoModelForImageClassification_inference.py) — inference.
- `requirements.hf39pt.pip.txt` — dependencies.
