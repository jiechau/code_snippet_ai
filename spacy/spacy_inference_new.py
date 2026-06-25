
import spacy
best_model = spacy.load(r"./output/model-last") #load the best model
new_doc = best_model("台灣是一個位於亞洲東部的島嶼國家。")
for ent in new_doc.ents:
    print(f"{ent.text} ({ent.label_})")
