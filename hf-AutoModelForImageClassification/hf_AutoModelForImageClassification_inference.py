# -*- coding: utf-8 -*-
"""
use hf39pt
"""

# https://huggingface.co/docs/transformers/tasks/image_classification
# use hf39pt

# !pip install transformers datasets evaluate
 
is_colab = False
is_rog = False # that C:/ problem， and gram problem

# vars
LOOP_NUM = 1
EPOCHS = 10 # 2
DATASET_NUM = 100
TRAIN_SET_RATIO = 0.9
TEST_size = 0.1
BATCH_SIZE = 32
batch_size = BATCH_SIZE
LEARNING_RATE = 5e-5 # 3e-5 # 2e-4 # 1e-5
WEIGHT_DECAY_RATE = 0.01
NUM_WARMUP_STEPS = 0
WARMUP_RATIO = 0.1

MY_EVAL_STEPS = 100
MY_SAVE_STEPS = 100
MY_SAVE_EPOCHS = 10

#is_first_time_create_model = True
is_first_time_create_model = False

is_load_hf_from_local = True
#is_load_hf_from_local = False

#is_save_hf_to_local = True
is_save_hf_to_local = False

saved_model_suffix = "-pt-trainer"

#model_name = "facebook/convnextv2-tiny-22k-224"
#model_name = "facebook/convnextv2-base-22k-224"
#model_name = "microsoft/swin-tiny-patch4-window7-224"
#model_name = "microsoft/swin-small-patch4-window7-224"
model_name = "microsoft/swin-base-patch4-window7-224-in22k"
#model_name = "google/vit-base-patch16-224"
#model_name = "google/vit-base-patch16-224-in21k"

#saved_model_name = "microsoft-swin-base-patch4-window7-224-in22k-pt-trainer"

import warnings
warnings.filterwarnings("ignore")
#warnings.filterwarnings("default")

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


# 新增 def get_hf_ns_model_base(_model_name)

import os
# is_load_hf_from_local = True
#saved_dir = '/content/saved' # this is just for test
#saved_dir = colab_dir + '/_saved_model/huggingface/transformers'
hf_local_saved_dir = colab_dir + '/_saved_model/huggingface/transformers'
os.environ["hf_local_saved_dir"] = hf_local_saved_dir
os.makedirs(hf_local_saved_dir, exist_ok=True)

# this is for save/load hugging face default pretrained model
def get_hf_ns_model_base(_model_name):
    if '/' in _model_name:
        # 'yentinglin/bert-base-zhtw'
        # ref
        ns_this, model_this = _model_name.split('/')
    else:
        # 'distilbert-base-multilingual-cased'
        # ref
        ns_this = ''
        model_this = _model_name
    # return
    return (ns_this, model_this)

def get_hf_sub_dir(_model_name):
    if '/' in _model_name:
        # 'yentinglin/bert-base-zhtw'
        # ref
        ns_this, model_this = _model_name.split('/')
        hub_ref = '/hub/models--' + ns_this + '--' + model_this
        # saved
        sub_dir = '/' + ns_this + '/' + model_this
    else:
        # 'distilbert-base-multilingual-cased'
        # ref
        model_this = _model_name
        hub_ref = '/hub/models--' + model_this
        # saved
        sub_dir = '/' + model_this
    # return
    return (hub_ref, sub_dir)

def load_hf_pretrained(_model_name):
    hub_ref, sub_dir = get_hf_sub_dir(_model_name)
    # saved
    model_dir_saved = hf_local_saved_dir + sub_dir
    print(model_dir_saved)
    # should check if existed
    # if fail, return False
    pass
    os.environ["model_dir_saved"] = model_dir_saved
    #
    return model_dir_saved

def save_hf_pretrained(_model_name):
    hub_ref, sub_dir = get_hf_sub_dir(_model_name)
    # ref
    model_dir_ref = HF_HOME + hub_ref
    os.environ["model_dir_ref"] = model_dir_ref
    # saved
    model_dir_saved = hf_local_saved_dir + sub_dir
    print(model_dir_saved)
    os.environ["model_dir_saved"] = model_dir_saved
    os.makedirs(model_dir_saved, exist_ok=True)
    # do the copy
    #!cp -L $model_dir_ref/snapshots/*/* $model_dir_saved/
    import subprocess
    try:
        command = f"cp -L {model_dir_ref}/snapshots/*/* {model_dir_saved}/"
        subprocess.run(command, shell=True, check=True)
        print("Files copied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")



from datetime import datetime,timedelta
_t1 = datetime.now()
def msg_time():
    global _t1
    _t2 = datetime.now()
    time_diff = (_t2 - _t1).total_seconds()
    time_diff_float_min = round(time_diff / 60, 3)
    time_diff_float = round(time_diff, 3)
    #msg_all = _t2.strftime("%Y-%m-%d %H:%M:%S") + ' ' + str(time_diff_float_min) + ' min'
    msg_all = _t2.strftime("%Y-%m-%d %H:%M:%S") + ' ' + str(time_diff_float) + ' sec'
    _t1 = _t2
    return msg_all
    #return




# load pretrained

#model_name = "microsoft/swin-base-patch4-window7-224-in22k" # pre-trained model from which to fine-tune
#saved_model_suffix = "-pt-trainer"
#saved_model_name = "microsoft-swin-base-patch4-window7-224-in22k-pt-trainer"
_ns, _model_base = get_hf_ns_model_base(model_name)
if _ns == '':
    saved_model_name = _model_base + saved_model_suffix
else:
    saved_model_name = _ns + '-' + _model_base + saved_model_suffix


from transformers import AutoModelForImageClassification
# is_save_hf_to_local
if is_save_hf_to_local:
    model_save = AutoModelForImageClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    save_hf_pretrained(model_name)
    checkpoint_hf = load_hf_pretrained(model_name)
    from transformers import AutoImageProcessor
    image_processor_save = AutoImageProcessor.from_pretrained(model_name)
    image_processor_save.save_pretrained(checkpoint_hf)

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

print(checkpoint)


aa = '''
# Preprocess
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
from transformers import AutoModelForImageClassification, AutoImageProcessor
model = AutoModelForImageClassification.from_pretrained(checkpoint)
'''


# inference
product_image_url = 'https://aaa.com/aaa.jpg'
product_image_url = 'https://aaa.com/aaa.webp'

# 1
# classifier
from transformers import pipeline
classifier = pipeline("image-classification", model=model_save_h5_dir)
result_list = classifier(product_image_url)
print(result_list[0])
print('--' * 20)

# 2
# logits = model(**inputs).logits
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained(model_save_h5_dir)
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained(model_save_h5_dir)

import urllib
import numpy as np
from PIL import Image
import io
with urllib.request.urlopen(product_image_url) as url:
    product_image_data = url.read()
    predict_img_arr = np.array(Image.open(io.BytesIO(product_image_data)))

inputs = image_processor(predict_img_arr, return_tensors="pt")
logits = model(**inputs).logits
import torch
predicted_class_id = torch.argmax(logits, dim=-1).item()
#predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
print('')
print('predicted:', predicted_class_id)
print('predicted:', model.config.id2label[predicted_class_id])

print('')
import torch
values, indices = torch.topk(logits, k=10, dim=-1)
#values, indices = tf.nn.top_k(logits, k=10)
values_np = values.detach().numpy().flatten()
#values_np = values.numpy().flatten()
indices_np = indices.numpy().flatten()
for index, value in zip(indices_np, values_np):
    print(f"{index}: {value} {model.config.id2label[index]}")



