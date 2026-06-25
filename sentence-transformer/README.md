# Sentence-Transformers

Examples using [`sentence-transformers`](https://www.sbert.net/) to produce sentence
embeddings and compute similarity.

## Setup

```bash
/usr/local/bin/python3.9 -m venv --system-site-packages sss
source sss/bin/activate
pip install --upgrade pip
pip install -r pt_sentencetransformer.requirements.cuda.pip.txt   # GPU
# or CPU:
pip install -r pt_sentencetransformer.requirements.cpu.pip.txt
```

## Run

```bash
python pt_sentencetransformer1.py
python pt_sentencetransformer2.py
python pt_sentencetransformer3.py
```

## Files

- [pt_sentencetransformer1.py](pt_sentencetransformer1.py), [pt_sentencetransformer2.py](pt_sentencetransformer2.py), [pt_sentencetransformer3.py](pt_sentencetransformer3.py) — embedding / similarity examples.
- `pt_sentencetransformer.requirements.cuda.pip.txt` / `pt_sentencetransformer.requirements.cpu.pip.txt` — GPU / CPU dependencies.
