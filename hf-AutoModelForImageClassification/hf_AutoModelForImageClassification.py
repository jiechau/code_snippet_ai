# -*- coding: utf-8 -*-
"""
use hf39pt
"""

# https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb
# use hf39pt

# get_ipython().system('pip install -q datasets transformers accelerate evaluate')

is_colab = False
is_rog = False # that C:/ problem， and gram problem

# vars
LOOP_NUM = 1
EPOCHS = 2 # 2
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

is_first_time_create_model = False
#is_first_time_create_model = True

is_load_hf_from_local = True
#is_load_hf_from_local = False

is_save_hf_to_local = True
#is_save_hf_to_local = False

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
    time_diff_float = round(time_diff / 60, 2)
    msg_all = _t2.strftime("%Y-%m-%d %H:%M:%S") + ' ' + str(time_diff_float) + ' min'
    _t1 = _t2
    return msg_all
    #return



# the normal example EuroSAT.zip
aa = '''
from datasets import load_dataset

# load a custom dataset from local/remote files or folders using the ImageFolder feature
!wget --no-check-certificate https://madm.dfki.de/files/sentinel/EuroSAT.zip
dataset = load_dataset("imagefolder", data_files="EuroSAT.zip")

# option 1: local/remote files (supporting the following formats: tar, gzip, zip, xz, rar, zstd)
#dataset = load_dataset("imagefolder", data_files="https://madm.dfki.de/files/sentinel/EuroSAT.zip")

# note that you can also provide several splits:
# dataset = load_dataset("imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]})

# note that you can push your dataset to the hub very easily (and reload afterwards using load_dataset)!
# dataset.push_to_hub("nielsr/eurosat")
# dataset.push_to_hub("nielsr/eurosat", private=True)

# option 2: local folder
# dataset = load_dataset("imagefolder", data_dir="path_to_folder")

# option 3: just load any existing dataset from the hub, like CIFAR-10, FashionMNIST ...
# dataset = load_dataset("cifar10")


# split up training into training + validation
splits = dataset["train"].train_test_split(test_size=0.1)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

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

aa = msg_time()
df_all = df_all[:DATASET_NUM].copy()
df_all['image'] = df_all.apply(lambda r: load_image_as_byte_array(r.img_location), axis=1)
df_all = df_all[['image','label']].copy()
print(msg_time(), 'done prepare df:', len(df_all))
import datasets
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
features = Features({
    'image': datasets.Image(decode=True, id=None),
    'label': ClassLabel(num_classes=len(label2id), names=list(label2id.keys()), names_file=None, id=None)
})
train_dataset = Dataset.from_pandas(df_all, features=features, preserve_index=False)
print(msg_time(), 'done prepare dataset')

# split up training into training + validation
splits = train_dataset.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']




# load pretrained

#model_name = "microsoft/swin-base-patch4-window7-224-in22k" # pre-trained model from which to fine-tune
#saved_model_suffix = "-pt-trainer"
#saved_model_name = "microsoft-swin-base-patch4-window7-224-in22k-pt-trainer"
_ns, _model_base = get_hf_ns_model_base(model_name)
if _ns == '':
    saved_model_name = _model_base + saved_model_suffix
else:
    saved_model_name = _ns + '-' + _model_base + saved_model_suffix
print(model_name)
print(saved_model_name)

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

# Preprocess
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained(checkpoint)


# In[ ]:
#import sys
#sys.exit()

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


# Next, we can preprocess our dataset by applying these functions. We will use the `set_transform` functionality, which allows to apply the functions above on-the-fly (meaning that they will only be applied when the images are loaded in RAM).
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
model = AutoModelForImageClassification.from_pretrained(
    #model_checkpoint,
    checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

args = TrainingArguments(
    #f"abcabcabc-finetuned-eurosat", # 這個名字也是自己取的
    model_save_h5_dir,
    remove_unused_columns=False,
    evaluation_strategy = "no", #"epoch",
    save_strategy = "no", #  "epoch",
    learning_rate=LEARNING_RATE, # 5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    #logging_dir=tmp_dir + "/AutoModelForImageClassification",
    #logging_steps=10,
    #load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)
print(model_name) # swin-base-patch4-window7-224-in22k
print(model_save_h5_dir)


from evaluate import load
metric = load("accuracy")
#metric = load("accuracy", cache_dir="/content/tmp")

import numpy as np
# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

import torch
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

#import sys
#sys.exit()
image_processor.save_pretrained(model_save_h5_dir)
model.save_pretrained(model_save_h5_dir, safe_serialization=True)

train_results = trainer.train()
trainer.log_metrics("train", train_results.metrics)
# rest is optional but nice to have
#trainer.save_model()
#trainer.log_metrics("train", train_results.metrics)
#trainer.save_metrics("train", train_results.metrics)
#trainer.save_state()
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)




