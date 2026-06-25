# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "FlagEmbedding",
#   "torch",
# ]
# ///

# use inline uv (no need pyproject.toml uv.lock)
# uv run bge_m3.py

# dense + sparse + colbert embeddings demo
# add custom token

import torch
import torch.nn.functional as F
from FlagEmbedding import BGEM3FlagModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=device)


def get_embeddings(text):
    output = model.encode(
        text,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    mapped = {
        model.tokenizer.convert_ids_to_tokens(int(tid)): float(w)
        for tid, w in output['lexical_weights'].items()
    }
    return {
        'dense': output['dense_vecs'],
        'colbert': output['colbert_vecs'],
        'sparse': sorted(mapped.items(), key=lambda x: x[1], reverse=True),
    }


sentences = [
    '【Apple】iPhone 17 Pro Max 2TB (5G) 手機',
    '羅技無線滑鼠 MX Anywhere 3S',
]

for text in sentences:
    result = get_embeddings(text)
    print(f"\nSentence: {text}")
    print(f"  Dense shape:   {result['dense'].shape}")
    print(f"  ColBERT shape: {result['colbert'].shape}")
    print(f"  Top 10 sparse tokens:")
    for token, weight in result['sparse'][:10]:
        print(f"    {token}: {weight:.4f}")
    print("-" * 50)


# --- Custom token: warm-start '羅技' ---
num_added = model.tokenizer.add_tokens(['羅技'])
model.model.model.resize_token_embeddings(len(model.tokenizer), mean_resizing=False)

id_luo = model.tokenizer.convert_tokens_to_ids('羅')
id_ji  = model.tokenizer.convert_tokens_to_ids('技')
id_new = model.tokenizer.convert_tokens_to_ids('羅技')

with torch.no_grad():
    emb = model.model.model.embeddings.word_embeddings.weight
    emb[id_new] = (emb[id_luo] + emb[id_ji]) / 2

sentence = '羅技無線滑鼠 MX Anywhere 3S'
output = model.encode(sentence, return_sparse=True)
mapped = {
    model.tokenizer.convert_ids_to_tokens(int(k)): float(v)
    for k, v in output['lexical_weights'].items()
}
sorted_w = sorted(mapped.items(), key=lambda x: x[1], reverse=True)

print(f"\nCustom token '羅技' (ID {id_new}) warm-start result:")
for token, weight in sorted_w:
    marker = " <-- activated" if token == '羅技' else ""
    print(f"  {token}: {weight:.6f}{marker}")
