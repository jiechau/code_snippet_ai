# BGE-M3

BAAI's [BGE-M3](https://huggingface.co/BAAI/bge-m3) embedding demo using
[`FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding) (its dev tool).

Shows the three embedding outputs BGE-M3 produces — **dense**, **sparse** (lexical
weights), and **ColBERT** (multi-vector) — plus how to add a custom token.

## Run

This script uses an **inline uv** script header (the `# /// script` block at the top),
so there is no need for a `pyproject.toml` or `uv.lock` — uv resolves dependencies on
the fly:

```bash
uv run bge_m3.py
```

## Files

- [bge_m3.py](bge_m3.py) — dense + sparse + ColBERT embedding demo, with a custom token.
