#
# python pt_sentencetransformer1.py 
# python pt_sentencetransformer1.py cpu
#
DATASET_NUM = 10_000
import sys
import torch
print('***torch.cuda.is_available()', torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if len(sys.argv) == 2 and sys.argv[1] == 'cpu':
    device = 'cpu'
print('use device', device)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
#model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')

#Our sentences we like to encode
s = 'This framework generates embeddings for each input sentence'
sentences = [s for i in range(DATASET_NUM)]
print(len(sentences))

import time
start = time.time()
#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)
#
end = time.time()
print(end - start, 'sec')
print(device)


#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    #print("Sentence:", sentence)
    #print("Embedding:", embedding)
    #print("")
    pass
