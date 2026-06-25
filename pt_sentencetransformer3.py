#
# python pt_sentencetransformer3.py 
# python pt_sentencetransformer3.py cpu
#
DATASET_NUM = 1_000
import sys
import torch
print('***torch.cuda.is_available()', torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if len(sys.argv) == 2 and sys.argv[1] == 'cpu':
    device = 'cpu'
print('use device', device)

from torchvision import datasets
train_data = datasets.FashionMNIST(
    root='/tmp/tmp/data', 
    train=True,
    download=True
)

# Get a random sample of 10 images 
import numpy as np    
rnd_inds = np.random.randint(0, len(train_data), DATASET_NUM)
# images = [np.array(train_data[i][0]) for i in rnd_inds] # 
images = [train_data[i][0] for i in rnd_inds] # already <class 'PIL.Image.Image'>
labels = [train_data[i][1] for i in rnd_inds]

'''
# Display the images 
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 10, figsize=(20,2))
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(labels[i])
    ax.axis('off')
plt.show()
'''

# https://www.sbert.net/examples/applications/paraphrase-mining/README.html

from sentence_transformers import SentenceTransformer, util
#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32', device=device)

import time
start = time.time()
paraphrases = util.paraphrase_mining(model, images)
end = time.time()

# print(paraphrases)
for paraphrase in paraphrases:
    score, i, j = paraphrase
    print("{} \t\t {} \t\t Score: {:.4f}".format(i, j, score))
print(labels)

print(end - start, 'sec')
print(device)





