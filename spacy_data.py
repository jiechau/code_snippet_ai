training_data = [
    ("台灣的風景都非常美麗",[(0,2,"GPE")]),
    ("亞洲東部國家的一些特色包括擁有豐富多彩的文化遺產和傳統習俗。",[(0,4,"LOC"),(4,6,"ORG")]),
    ("在亞洲東部國家，食品文化有著獨特的地位，其美食吸引著眾多遊客前來品嚐。",[(1,5,"LOC"),(5,7,"ORG")]),
    ("亞洲東部國家的經濟以製造業和出口為主，是全球經濟中的關鍵角色。",[(0,4,"LOC"),(4,6,"ORG")]),
    ("在亞洲東部國家，教育非常重要，其高水平的教育體系吸引著世界各地的學生前來留學。",[(3,5,"LOC"),(5,7,"ORG")]),
    # ("亞洲東部國家的一些城市，如東京、首爾和上海等，擁有先進的科技和發達的城市建設，是現代化的代表。",[(2,4,"LOC"),(4,6,"ORG"),(13,15,"ORG"),(16,18,"ORG"),(19,21,"ORG")]),
]

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.blank("zh")
# the DocBin will store the example documents
db = DocBin()
for text, annotations in tqdm(training_data):
    _doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        span = _doc.char_span(start, end, label=label)
        ents.append(span)
    _doc.ents = ents
    db.add(_doc)
db.to_disk("./train.spacy")
db.to_disk("./dev.spacy") # for now
