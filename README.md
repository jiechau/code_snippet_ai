# check_sys_info — AI / ML Code Snippets

A collection of small, self-contained AI / ML code snippets and templates — handy for
checking that a machine's Python / TensorFlow / PyTorch / CUDA setup works, and as
starting points for common tasks (embeddings, classification, regression, NER, RAG,
fine-tuning).

Each snippet lives in its own folder with its own `README.md` describing what it does
and how to run it. Folders are self-contained: they ship the `requirements.*.pip.txt`
file(s) they need.

```bash
git clone https://gitlab.com/jiechau/check_sys_info.git
```

See [venv-setup/](venv-setup/) for the general virtual-environment pattern shared by
most snippets.

## Contents

### Embeddings & similarity
- [BGE-M3/](BGE-M3/) — BAAI BGE-M3 dense + sparse + ColBERT embedding demo (run with inline `uv`).
- [sentence-transformer/](sentence-transformer/) — sentence embeddings & similarity with `sentence-transformers`.

### Regression
- [tensorflow-regression/](tensorflow-regression/) — DNN regression in TensorFlow/Keras (+ a TensorFlow.js browser export).
- [torch-regression/](torch-regression/) — DNN regression in PyTorch.

### Image classification
- [tensorflow-image-classification/](tensorflow-image-classification/) — image classification in TensorFlow/Keras.
- [torch-image-classification/](torch-image-classification/) — image classification in PyTorch (GPU/CPU + FLOPs).
- [hf-TFAutoModelForImageClassification/](hf-TFAutoModelForImageClassification/) — Hugging Face image classification (TensorFlow backend).
- [hf-AutoModelForImageClassification/](hf-AutoModelForImageClassification/) — Hugging Face image classification (PyTorch backend).

### Text / NLP
- [hf-TFDistilBertForSequenceClassification/](hf-TFDistilBertForSequenceClassification/) — Hugging Face text sequence classification (TensorFlow backend).
- [spacy/](spacy/) — Chinese NER with spaCy (build data, train, infer).

### LLM / RAG
- [hf-llama2-NousResearch/](hf-llama2-NousResearch/) — fine-tune Llama 2 (NousResearch) with Hugging Face (train → merge → infer).
- [graphrag_book/](graphrag_book/) — Microsoft GraphRAG over a book using OpenAI `gpt-4o-mini`.

### Environment & infra
- [venv-setup/](venv-setup/) — Python venv pattern and the list of environments used across the repo.
- [env-check/](env-check/) — scripts to check Python / TF / PyTorch / CUDA / GPU.
- [docker/](docker/) — running the snippets in TensorFlow / PyTorch GPU containers.
- [hardware-notes/](hardware-notes/) — machines, NVIDIA drivers, and CUDA/cuDNN version notes.
