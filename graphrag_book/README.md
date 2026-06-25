# Microsoft GraphRAG

A note/example of running [Microsoft GraphRAG](https://microsoft.github.io/graphrag/)
over a book. In 2024/07 OpenAI launched `gpt-4o-mini`, which is cheap enough to make
this practical.

Tutorial: <https://microsoft.github.io/graphrag/posts/get_started/>

## Setup

1. Activate your OpenAI **Organization account** (not a monthly user subscription) by
   depositing $10: <https://platform.openai.com/settings/organization/billing/overview>
2. Create an API key: <https://platform.openai.com/organization/api-keys>
3. Create a virtual env, e.g. `py312graphrag` (see [requirements.py312graphrag.pip.txt](requirements.py312graphrag.pip.txt)).
4. You'll need access to OpenAI's chat **and** embeddings APIs.
5. Get a `.txt` source (e.g. *Alice's Adventures in Wonderland*) and estimate tokens
   with <https://platform.openai.com/tokenizer>.

## Workflow

```bash
mkdir -p ./ragtest_ch/input
cp xxx ./ragtest_ch/input
python -m graphrag.index --init --root ./ragtest_ch
vi ragtest_ch/.env                 # put API key
vi ragtest_ch/settings.yaml        # model: gpt-4o-mini
python -m graphrag.index --root ./ragtest_ch
python -m graphrag.query --root ./ragtest_ch --method global "這本書的主旨是什麼?"
python -m graphrag.query --root ./ragtest_ch --method global "這篇論文主要在說明什麼?"
python -m graphrag.query --root ~/ragtest_en --method global "what is main idea of this book?"
```

## Files

- [ragtest_az/settings.yaml](ragtest_az/settings.yaml) — example settings (Azure OpenAI).
- [ragtest_openai/settings.yaml](ragtest_openai/settings.yaml) — example settings (OpenAI).
- `Alices-Adventures-in-Wonderland.{txt,pdf}`, `Through-the-Looking-Glass.{txt,pdf}` — sample source books.
- `requirements.py312graphrag.pip.txt` — dependencies (Python 3.12).
