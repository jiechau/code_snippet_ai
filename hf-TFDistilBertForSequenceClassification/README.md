# Hugging Face — TFDistilBertForSequenceClassification

Text sequence classification with Hugging Face Transformers using the TensorFlow
`TFDistilBertForSequenceClassification` model, with a separate inference script.

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
python hf_TFDistilBertForSequenceClassification.py            # train
python hf_TFDistilBertForSequenceClassification_inference.py  # inference
```

## Files

- [hf_TFDistilBertForSequenceClassification.py](hf_TFDistilBertForSequenceClassification.py) — train.
- [hf_TFDistilBertForSequenceClassification_inference.py](hf_TFDistilBertForSequenceClassification_inference.py) — inference.
- `requirements.hf39tf.pip.txt` — dependencies.
