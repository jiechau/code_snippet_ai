# Python venv Setup

Most snippets in this repo run in a Python virtual environment created with
`--system-site-packages` (so system CUDA / driver packages are visible) and a
per-snippet `requirements.*.pip.txt` file. Each snippet folder ships the
requirements file(s) it needs.

## General pattern

```bash
/usr/local/bin/python3.9 -m venv --system-site-packages <env-name>
source <env-name>/bin/activate
pip install --upgrade pip
pip install -r requirements.<env-name>.pip.txt
```

## Environments used across the repo

| env       | Python | Used by |
| --------- | ------ | ------- |
| `tf39`    | 3.9    | TensorFlow regression / image classification (GPU) |
| `tf39cpu` | 3.9    | TensorFlow regression / image classification (CPU) |
| `pt39`    | 3.9    | PyTorch regression / image classification (GPU or CPU) |
| `pt311`   | 3.11   | PyTorch regression / image classification (GPU or CPU) |
| `sss`     | 3.9    | sentence-transformers |
| `py39spacy` | 3.9  | spaCy |
| `hf39tf`  | 3.9    | Hugging Face (TensorFlow backend) |
| `hf39pt`  | 3.9    | Hugging Face (PyTorch backend) |
| `ppp`     | 3.9    | Llama 2 fine-tuning |
| `py312graphrag` | 3.12 | GraphRAG |

## uv (alternative)

Some standalone scripts use [uv](https://github.com/astral-sh/uv) with an inline
script header instead of a venv — see [../BGE-M3/](../BGE-M3/) for an example
(`uv run bge_m3.py`, no `pyproject.toml` / `uv.lock` needed).
