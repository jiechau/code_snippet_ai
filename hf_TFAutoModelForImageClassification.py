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

is_save_hf_to_local = True
#is_save_hf_to_local = False

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

# if this is the very first time
# you don't even have base model
import os
hf_local_saved_dir = colab_dir + '/_saved_model/huggingface/transformers'
os.environ["hf_local_saved_dir"] = hf_local_saved_dir
os.makedirs(hf_local_saved_dir, exist_ok=True)

# this is for save/load hugging face default pretrained model
def get_hf_sub_dir(_model_name):
    if '/' in _model_name:
        # 'google/vit-base-patch16-224-in21k'
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
    time_diff_float = round(time_diff / 60, 2)
    msg_all = _t2.strftime("%Y-%m-%d %H:%M:%S") + ' ' + str(time_diff_float) + ' min'
    _t1 = _t2
    return msg_all
    #return



# the normal example
aa = '''
# load from internet, 10 min
#from datasets import load_dataset
#food = load_dataset("food101", split="train[:5000]")
#
# Save the dataset to a specified directory
#food101_5000 = load_dataset("food101", split="train[:5000]")
#dir_food101 = dir_all_data + '/food101_5000'
#food101_5000.save_to_disk(dir_food101)
#type(food101_5000) # datasets.arrow_dataset.Dataset
#
# or
#
# Load the dataset from the specified directory
from datasets import load_from_disk
dir_food101 = dir_all_data + '/food101_5000'
food = load_from_disk(dir_food101)

# then
food = food.train_test_split(test_size=TEST_size)
'''

# 
# or
# use your own dataset to train
#
# dummy data
# df_all (is all we need)
import sys
import pandas as pd
import numpy as np
from PIL import Image
dummy_image_path = tmp_dir + '/tmp_image.jpg'
dummy_image = Image.new('RGB', (384, 512))
dummy_image.save(dummy_image_path)
df_all = pd.DataFrame({
    'img_location': [dummy_image_path] * 5000, 
    'label': np.random.randint(1, 101, size=5000)  
})
# dummy label
# df_label is all we need
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




# start

import io
import PIL
from PIL import Image
def load_image_as_byte_array(image_path):
    with PIL.Image.open(image_path) as image:
        img_byte_arr = io.BytesIO()
        #image.save(img_byte_arr, format='JPEG')  # You can change the format if needed
        image.save(img_byte_arr, format=image.format)
        img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

#df = df_all[:DATASET_NUM].copy()
df = df_all.sample(frac=1).reset_index(drop=True)[:DATASET_NUM].copy()
df['image'] = df.apply(lambda r: load_image_as_byte_array(r.img_location), axis=1)
df = df[['image','label']]

# Convert the DataFrame to a Dataset
import datasets
from datasets import Dataset, Features, ClassLabel, Image
features = Features({
    'image': datasets.Image(decode=True, id=None),
    'label': ClassLabel(num_classes=len(label2id), names=list(label2id.keys()), names_file=None, id=None)
})
food = Dataset.from_pandas(df, features=features, preserve_index=False)

# Optionally, set the format to torch (PyTorch), tensorflow, etc., if you plan to use it with those frameworks
# food_dataset.set_format(type='torch', columns=['image', 'category_id'])

# split data
food = food.train_test_split(test_size=TEST_size)

print(msg_time(), 'done prepare dataset')




#model_name = "google/vit-base-patch16-224-in21k"
#saved_model_name = "google-vit-base-patch16-224-in21k-tf-keras"

# model
# is_save_hf_to_local
if is_save_hf_to_local:
    from transformers import TFAutoModelForImageClassification
    model_save = TFAutoModelForImageClassification.from_pretrained(
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

# Preprocess
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
#size = (image_processor.size["height"], image_processor.size["width"])
#size # (224, 224)

from tensorflow import keras
from tensorflow.keras import layers
size = (image_processor.size["height"], image_processor.size["width"])
train_data_augmentation = keras.Sequential(
    [
        layers.RandomCrop(size[0], size[1]),
        layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="train_data_augmentation",
)
val_data_augmentation = keras.Sequential(
    [
        layers.CenterCrop(size[0], size[1]),
        layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ],
    name="val_data_augmentation",
)

import numpy as np
import tensorflow as tf
import PIL
from PIL import Image
def convert_to_tf_tensor(image: PIL.Image):
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    # `expand_dims()` is used to add a batch dimension since
    # the TF augmentation layers operates on batched inputs.
    return tf.expand_dims(tf_image, 0)
def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    images = [
        train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch
def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch
food["train"].set_transform(preprocess_train)
food["test"].set_transform(preprocess_val)


# the DefaultDataCollator does not apply additional preprocessing, such as padding.
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator(return_tensors="tf")
# converting our train dataset to tf.data.Dataset
tf_train_dataset = food["train"].to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)
# converting our test dataset to tf.data.Dataset
tf_eval_dataset = food["test"].to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)


# Evaluate
import evaluate
accuracy = evaluate.load("accuracy")
import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Train
from transformers import create_optimizer
optimizer, lr_schedule = create_optimizer(
    init_lr=LEARNING_RATE,
    num_train_steps=len(food["train"]) * EPOCHS,
    weight_decay_rate=WEIGHT_DECAY_RATE,
    num_warmup_steps=NUM_WARMUP_STEPS,
)
from transformers import TFAutoModelForImageClassification
model = TFAutoModelForImageClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)



# train
print(msg_time(), 'start')
for iii in range(LOOP_NUM):

    from transformers.keras_callbacks import KerasMetricCallback
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)
    # train
    history = model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=EPOCHS, callbacks=[metric_callback])
    #model_name = "google/vit-base-patch16-224-in21k"
    #saved_model_name = "google-vit-base-patch16-224-in21k-tf-keras"
    model_save_h5_dir = dir_all_model + '/' + saved_model_name
    os.makedirs(model_save_h5_dir, exist_ok=True)
    # save
    image_processor.save_pretrained(model_save_h5_dir)
    model.save_pretrained(model_save_h5_dir, safe_serialization=True)
    #print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(msg_time(), 'loop:' + str(iii))











