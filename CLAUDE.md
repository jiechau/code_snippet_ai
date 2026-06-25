# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of small, **self-contained** AI / ML code snippets and templates (embeddings,
classification, regression, NER, RAG, fine-tuning), plus tools for verifying a machine's
Python / TensorFlow / PyTorch / CUDA setup. There is no central build, package, test suite,
or CI — each snippet folder is independent and ships its own dependencies.

## Structure & conventions

Each snippet lives in its own top-level folder with:
- a `README.md` describing what it does, which env to use, and how to run it,
- one or more `requirements.<env-name>.pip.txt` files it needs,
- the actual script(s).

Folders are intentionally self-contained — do **not** hoist shared code or a top-level
requirements file across snippets. When adding or editing a snippet, keep its dependencies
in its own `requirements.*.pip.txt` and update that folder's `README.md`. If you add a new
top-level snippet folder, also add it to the **Contents** list in the root [README.md](README.md).

The filename middle segment (`tf39`, `pt39`, `hf39pt`, …) names the virtual environment the
snippet expects — it must match an entry in the env table in [venv-setup/README.md](venv-setup/README.md).

## Running snippets — two patterns

**1. venv (most snippets).** Created with `--system-site-packages` so system CUDA / driver
packages are visible, plus a per-snippet requirements file:

```bash
/usr/local/bin/python3.9 -m venv --system-site-packages <env-name>
source <env-name>/bin/activate
pip install --upgrade pip
pip install -r requirements.<env-name>.pip.txt
python <script>.py
```

Env names (Python version → used by) are tabulated in [venv-setup/README.md](venv-setup/README.md):
`tf39`/`tf39cpu` (TF GPU/CPU), `pt39`/`pt311` (PyTorch), `sss` (sentence-transformers),
`py39spacy` (spaCy), `hf39tf`/`hf39pt` (Hugging Face TF/PyTorch backends), `ppp` (Llama 2
fine-tuning), `py312graphrag` (GraphRAG). Many snippets ship both a GPU and a CPU
requirements file.

**2. inline uv (standalone scripts).** Some scripts embed a `# /// script` PEP-723 header
declaring `requires-python` and `dependencies`, so no venv / `pyproject.toml` / `uv.lock` is
needed — uv resolves on the fly. See [BGE-M3/bge_m3.py](BGE-M3/bge_m3.py):

```bash
uv run bge_m3.py
```

## Environment / hardware

- [env-check/](env-check/) — scripts to verify Python / TF / PyTorch / CUDA / GPU before running a snippet.
- [docker/](docker/) — running snippets inside official TensorFlow / PyTorch GPU containers.
- [hardware-notes/](hardware-notes/) — machine, NVIDIA driver, and CUDA/cuDNN version notes.

## Multi-step snippets

A few snippets are pipelines rather than a single script — follow the ordering in their
README:
- [spacy/](spacy/) — Chinese NER: build data → train → infer.
- [hf-llama2-NousResearch/](hf-llama2-NousResearch/) — Llama 2 fine-tune: train → merge → infer.
- [graphrag_book/](graphrag_book/) — Microsoft GraphRAG over a book (OpenAI `gpt-4o-mini`); contains separate `ragtest_az` / `ragtest_openai` configs.
