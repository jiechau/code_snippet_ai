# spaCy — NER (Chinese)

Named-entity recognition with [spaCy](https://spacy.io/), including building training
data, training a transformer-based Chinese NER model, and running inference.

## Setup

```bash
/usr/local/bin/python3.9 -m venv --system-site-packages py39spacy
source py39spacy/bin/activate
pip install --upgrade pip
pip install spacy
pip install spacy[transformers]
pip install cupy-cuda12x          # GPU only; check nvidia-smi / nvcc --version
python -m spacy download zh_core_web_trf
python -m spacy info
```

## Run

```bash
# inference with the pretrained model
python -W ignore spacy_inference.py

# build training data
python spacy_data.py

# train (see https://spacy.io/usage/training#config)
# zh, ner -> base_config.cfg
python -m spacy init fill-config ./base_config.cfg ./config.cfg
python -m spacy train config.cfg --output ./output \
  --paths.train ./train.spacy --paths.dev ./dev.spacy --gpu-id 0   # --training.max_epochs 10

# inference with the newly trained model
python -W ignore spacy_inference_new.py
```

## Files

- [spacy_data.py](spacy_data.py) — build `train.spacy` / `dev.spacy` training data.
- [base_config.cfg](base_config.cfg) — base training config (zh, ner).
- [spacy_inference.py](spacy_inference.py) — inference with the pretrained model.
- [spacy_inference_new.py](spacy_inference_new.py) — inference with the trained model.
- `requirements.py39spacy.pip.txt` — dependencies.
