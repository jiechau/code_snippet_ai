# Hugging Face — Llama 2 (NousResearch) Fine-tuning

Fine-tuning a Llama 2 model (NousResearch weights) with Hugging Face Transformers.
The workflow is split into three scripts so it can run on limited RAM/VRAM.

## Setup

Use the `ppp` virtual environment:

```bash
/usr/local/bin/python3.9 -m venv --system-site-packages ppp
source ppp/bin/activate
pip install --upgrade pip
pip install -r requirements.ppp.pip.txt
```

## Run (3 steps)

```bash
python hf_llama2_NousResearch.py            # 1. train
python hf_llama2_NousResearch_merge.py      # 2. merge adapter -> save custom model
python hf_llama2_NousResearch_inference.py  # 3. inference with the custom model
```

## Files

- [hf_llama2_NousResearch.py](hf_llama2_NousResearch.py) — train.
- [hf_llama2_NousResearch_merge.py](hf_llama2_NousResearch_merge.py) — merge after training, save custom model.
- [hf_llama2_NousResearch_inference.py](hf_llama2_NousResearch_inference.py) — inference with the custom model.
- `requirements.ppp.pip.txt` — dependencies.
