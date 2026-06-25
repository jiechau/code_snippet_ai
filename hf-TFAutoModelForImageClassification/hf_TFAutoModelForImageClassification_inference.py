# -*- coding: utf-8 -*-
"""
use hf39tf
"""

# https://huggingface.co/docs/transformers/tasks/image_classification
#!pip install transformers datasets evaluate

# vars
LOOP_NUM = 1
EPOCHS = 10
DATASET_NUM = 100 #
TRAIN_SET_RATIO = 0.9
TEST_size = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 3e-5 # 3e-5 2e-4 # 1e-5
WEIGHT_DECAY_RATE = 0.01
NUM_WARMUP_STEPS = 0

MY_EVAL_STEPS = 100
MY_SAVE_STEPS = 100
MY_SAVE_EPOCHS = 25


is_first_time_create_model = False
#is_first_time_create_model = True

is_load_hf_from_local = True
#is_load_hf_from_local = False

#is_save_hf_to_local = True
is_save_hf_to_local = False

model_name = "google/vit-base-patch16-224-in21k"
saved_model_name = "google-vit-base-patch16-224-in21k-tf-keras"
#model_name = "google/vit-base-patch16-224"
#saved_model_name = "google-vit-base-patch16-224-tf-keras"
#model_name = "microsoft/swin-small-patch4-window7-224"
#saved_model_name = "microsoft-swin-small-patch4-window7-224-tf-keras"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import os
colab_dir = '/tmp/tmp/Colab_Notebooks'
HF_HOME = '/tmp/tmp/transformers'
tmp_dir = '/tmp/tmp' # save tmp

# colab_dir
os.environ["colab_dir"] = colab_dir

# HF cache
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

# saved_model
dir_all_model = colab_dir + '/_saved_model'
os.environ["dir_all_model"] = dir_all_model

# _saved_data
dir_all_data = colab_dir + '/_saved_data'
os.environ["dir_all_data"] = dir_all_data

# tmp
os.environ["tmp_dir"] = tmp_dir
os.makedirs(tmp_dir, exist_ok=True)


from datetime import datetime,timedelta
_t1 = datetime.now()
def msg_time():
    global _t1
    _t2 = datetime.now()
    time_diff = (_t2 - _t1).total_seconds()
    time_diff_float = round(time_diff / 60, 2)
    msg_all = _t2.strftime("%Y-%m-%d %H:%M:%S") + ' ' + str(time_diff_float) + ' min'
    _t1 = _t2
    return msg_all
    #return




# dummy label
# df_label is all we need
import sys
import pandas as pd
import numpy as np
import random
import string
df_label = pd.DataFrame({
    'id_num': range(1, 101),  # Integer from 1 to 100
    'level_txt': [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 9))) for _ in range(100)]  # Corresponding random words
})
# make label/id
label2id, id2label = dict(), dict()
for index, row in df_label.iterrows():
    label2id[row['level_txt']] = str(row['id_num'])
    id2label[str(row['id_num'])] = row['level_txt']
label2id['default'] = '0'
id2label['0'] = 'default'
sorted_dict = dict(sorted(label2id.items(), key=lambda item: int(item[1])))
label2id = sorted_dict
sorted_dict = dict(sorted(id2label.items(), key=lambda item: int(item[0])))
id2label = sorted_dict



#model_name = "google/vit-base-patch16-224-in21k"
#saved_model_name = "google-vit-base-patch16-224-in21k-tf-keras"

# load model
model_save_h5_dir = dir_all_model + '/' + saved_model_name
if not os.path.exists(model_save_h5_dir):
    is_first_time_create_model = True

if is_first_time_create_model:
    checkpoint = model_name
    # if i want to load from local
    if is_load_hf_from_local:
        checkpoint = load_hf_pretrained(model_name)
else:
    checkpoint = model_save_h5_dir

# Preprocess
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
# model
from transformers import TFAutoModelForImageClassification
model = TFAutoModelForImageClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)



# inference

# 1
from transformers import pipeline
classifier = pipeline("image-classification", model=model_save_h5_dir)
product_image_url = 'https://aaa.com/aaa.jpg'
product_image_url = 'https://aaa.com/aaa.webp'
result = classifier(product_image_url)
for ii in result:
    print(ii)
aa = '''
import IPython
import requests
from io import BytesIO
response = requests.get(product_image_url)
IPython.display.display(IPython.display.Image(data=response.content))
print('')
'''



# 2
import io
import urllib
import numpy as np
import PIL
from PIL import Image
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained(model_save_h5_dir)
import tensorflow as tf
from transformers import TFAutoModelForImageClassification
model = TFAutoModelForImageClassification.from_pretrained(model_save_h5_dir)
product_image_url = 'https://aaa.com/aaa.jpg'
product_image_url = 'https://aaa.com/aaa.webp'
with urllib.request.urlopen(product_image_url) as url:
    product_image_data = url.read()
    predict_img_arr = np.array(PIL.Image.open(io.BytesIO(product_image_data)))
aa = '''
import matplotlib.pyplot as plt
plt.imshow(predict_img_arr)
plt.axis('off') # To not show axes for an image
plt.show()
'''
inputs = image_processor(predict_img_arr, return_tensors="tf")
logits = model(**inputs).logits
predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
print('')
print('predicted:', predicted_class_id)
print('predicted:', model.config.id2label[predicted_class_id])
print('')
values, indices = tf.nn.top_k(logits, k=10)
values_np = values.numpy().flatten()
indices_np = indices.numpy().flatten()
for index, value in zip(indices_np, values_np):
    print(f"{index}: {value} {model.config.id2label[index]}")







