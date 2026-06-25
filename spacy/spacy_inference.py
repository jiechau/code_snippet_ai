import spacy
nlp_zh = spacy.load('zh_core_web_trf')
doc = nlp_zh('台灣是一個位於亞洲東部的島嶼國家。')
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
