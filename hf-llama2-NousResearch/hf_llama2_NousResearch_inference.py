# 順序
# friDay_cate_id_hugging_face_llama2_NousResearch.ipynb
# friDay_cate_id_hugging_face_llama2_NousResearch_merge.ipynb
# friDay_cate_id_hugging_face_llama2_NousResearch_inference.ipynb

#colab T4
# 系統 RAM 2.4 / 12.7 GB
# GPU RAM 13.3 / 15.0 GB
# 執行模型的文本生成流程
#bigdata RTX 3060 12G
# 系統 RAM 14.0 / 16.0 GB
# GPU RAM 11753MiB / 12288MiB
# 2024-01-09 08:32:33 0.29 min start
# 2024-01-09 08:34:12 1.64 min done loading model
# 2024-01-09 08:34:13 0.03 min done loading pipeline
# 2024-01-09 08:40:28 6.24 min done inference # "<s>[INST] What is a large language model? [/INST]</s>"
# 2024-01-09 08:45:50 1.96 min done inference # "What is 2 + 2 equal to?"

# inference
# 其實 device_map="auto" 似乎無法控制，看運氣，每次執行不一樣
# 另一個變數是 prompt 的複雜度也影響很大
# bigdata 最後可以的那次，device_map="auto" 可以用滿 vram 和 ram
aaa = '''
$ export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
model = AutoModelForCausalLM.from_pretrained(model_save_h5_dir,
                         device_map="auto",
                         offload_folder="/tmp/offload",
                         torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_save_h5_dir,
                         torch_dtype=torch.float32) # TPU
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200) # 不能太大
'''

is_colab = False
is_TPU = False
is_rog = False # that C:/ problem， and gram problem

is_first_time_create_model = False
#is_first_time_create_model = True

is_load_hf_from_local = True

TRAIN_SET_RATIO = 0.99
EPOCHS = 1
DATASET_NUM = 100 #
#DATASET_NUM = -1 # bigdata
MAX_SEQ_LEN = 128
num_classes = 2_000


# 原先的 llama2
model_name = "NousResearch/Llama-2-7b-chat-hf"
# 算好的 peft
save_model_dir_name = 'nousresearch-llama-2-7b-chat-hf'
save_model_suffix = '-pt-qlora-peft'
# 最後合併好之後，要存的
custom_model_dir_name= 'nousresearch-llama-2-7b-custom'


# 這裡要注意 用 .py 以下都要 mark
#ccc = '''
# 安裝所需的 Python 套件
if is_colab:
    pass
    #!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers trl==0.4.7
    !pip install accelerate peft bitsandbytes trl

# colab time zone
if is_colab:
    pass
    !sudo ln -sf /usr/share/zoneinfo/Asia/Taipei /etc/localtime
    !sudo dpkg-reconfigure -f noninteractive tzdata
    !date

# NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968
if is_colab:
    import locale
    def getpreferredencoding(do_setlocale = True):
        return "UTF-8"
    locale.getpreferredencoding = getpreferredencoding
#'''

ddd = '''
# if TPU
if is_TPU:
    pass
    !pip install tensorflow==2.14
    # need restart runtime
# TPUs don't natively support the torch.float16
#model = model.to(torch.float32)
#tokenizer = tokenizer.to(torch.float32)
'''

import os
if is_colab:
    from google.colab import drive
    drive.mount('/content/drive')
    # colab_dir
    colab_dir = '/content/drive/MyDrive/Colab_Notebooks'
    # cached_dir
    HF_HOME = '/content/cache/huggingface/transformers'
    # tmp
    tmp_dir = '/content/tmp' # save tmp
else:
    if is_rog:
        colab_dir = 'C:/share/Colab_Notebooks'
        HF_HOME = 'C:/share/tmp/transformers'
        tmp_dir = 'C:/share/tmp/tmp' # save tmp

        # make the most of g-ram. fix gpu ram to 5120
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
            except RuntimeError as e:
                print(e)
    else:
        colab_dir = '/home/rundeck/_ai_bulk_files/Colab_Notebooks'
        HF_HOME = '/home/rundeck/_ai_bulk_files/cache/huggingface/transformers'
        tmp_dir = '/home/rundeck/_ai_bulk_files/tmp' # save tmp
# all

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

#!cd /content/drive/MyDrive/Colab_Notebooks/_saved_model/_saved_model_friDay/friDay_category_id \
#&& ls -l nousresearch-llama-2-7b-custom

dir_save_data = dir_all_data + '/friDay_data'
dir_saved_model = dir_all_model + '/_saved_model_friDay'
import pandas as pd
# df_all
pickle_location = dir_save_data + '/df_items_content_details_cate_20231213.pkl'
df_all = pd.read_pickle(pickle_location)
print('done', str(len(df_all)))
# df_all
pickle_location = dir_save_data + '/catg_content_20231213.pkl'
df_catg = pd.read_pickle(pickle_location)
# df_new
pickle_location = dir_save_data + '/df_items_content_details_cate_20240124.pkl'
df_latest = pd.read_pickle(pickle_location)
df_new = df_latest[~df_latest['pid'].isin(df_all['pid'])] # pid new
df_new = df_new[df_new['cate_id'].isin(df_catg['cate_id'])] # use old df_catg

print('done', str(len(df_catg)))
# max cate_id
print('max cate_id',str(df_all.cate_id.max()))
# df_new
print('done', str(len(df_new)))
print('max cate_id',str(df_new.cate_id.max()))


# 這裡通常 重新啟動執行階段

aaa = '''
# Empty VRAM
del model
del pipe
del trainer
import gc # 清理垃圾桶
gc.collect()
gc.collect()
'''

#!pip install accelerate==0.21.0

# 或是調整
# bigdata
if not is_colab:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:32'

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

#from google.colab import drive
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 原先的 llama2
model_name = "NousResearch/Llama-2-7b-chat-hf"
# 算好的 peft
save_model_dir_name = 'nousresearch-llama-2-7b-chat-hf'
save_model_suffix = '-pt-qlora-peft'
# 最後合併好之後，要存的
custom_model_dir_name= 'nousresearch-llama-2-7b-custom'


model_save_h5_dir = dir_all_model + '/_saved_model_friDay/friDay_category_id' + '/' + custom_model_dir_name # don't end '.h5'
os.environ["model_save_h5_dir"] = model_save_h5_dir
os.makedirs(model_save_h5_dir, exist_ok=True)
print('------------------------------------------------------------------')
print(model_save_h5_dir)
print('------------------------------------------------------------------')

print(msg_time(), 'start')

if is_colab:
    model = AutoModelForCausalLM.from_pretrained(model_save_h5_dir,
                            device_map="auto",
                            offload_folder=tmp_dir+"/offload",
                            torch_dtype="auto")
                            #torch_dtype=torch.float16) # GPU
                            #torch_dtype=torch.float32) # TPU
    tokenizer = AutoTokenizer.from_pretrained(model_save_h5_dir,
                            torch_dtype="auto")
                            #torch_dtype=torch.float16) # GPU
                            #torch_dtype=torch.float32) # TPU
else: # bigdata / smalldata
    model = AutoModelForCausalLM.from_pretrained(model_save_h5_dir,
                            device_map="auto",
                            offload_folder=tmp_dir+"/offload",
                            #torch_dtype="auto")
                            #torch_dtype=torch.float16) # GPU
                            torch_dtype=torch.float32) # TPU
    tokenizer = AutoTokenizer.from_pretrained(model_save_h5_dir,
                            #torch_dtype="auto")
                            #torch_dtype=torch.float16) # GPU
                            torch_dtype=torch.float32) # TPU
print(msg_time(), 'done loading model')

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
#pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200, device=-1)
#pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer, max_new_tokens= 2048)
print(msg_time(), 'done loading pipeline')

aa = msg_time()
prompt = "What is 2 + 2 equal to?"
#prompt = "What is a large language model?"
#prompt = "血型是 O 型的爸爸，和 血型是 B 型的媽媽，生出來的小孩會是什麼血型?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text']) # 輸出生成的文本
print(msg_time(), 'done inference')



def do_inference(_q):
    print(_q)
    aa = msg_time()
    print(msg_time(), 'start inference')
    result = pipe(_q)
    #result = pipe(f"<s>[INST] {_q} [/INST]")
    print(msg_time(), 'done inference')
    print(result[0]['generated_text'])
    print(' ')

# or random choice
def get_random(train_type):
    if train_type == 1:
        random_row = df_catg.sample(n=1)
        level_b_name = random_row['level_b_name'].iloc[0]
        cate_id = random_row['cate_id'].iloc[0]
        prompt = "請問這個 商品類別:'" + level_b_name + "' 的 '特殊類別編號' 是什麼?"
        print("prompt = \"" + prompt + "\" # " + str(cate_id))
        do_inference(prompt)
    if train_type == 2:
        random_row = df_all.sample(n=1)
        product_name = random_row['name'].iloc[0]
        level_b_name = random_row['level_b_name'].iloc[0]
        cate_id = random_row['cate_id'].iloc[0]
        prompt = "請問這個 商品描述:'" + product_name + "' 的 '商品類別' 是什麼?"
        print("prompt = \"" + prompt + "\" # " + level_b_name + "," + str(cate_id))
        do_inference(prompt)
    if train_type == 3:
        random_row = df_all.sample(n=1)
        product_name = random_row['name'].iloc[0]
        level_b_name = random_row['level_b_name'].iloc[0]
        cate_id = random_row['cate_id'].iloc[0]
        prompt = "請問這個 商品描述:'" + product_name + "' 的 '特殊類別編號' 是什麼?"
        print("prompt = \"" + prompt + "\" # " + level_b_name + "," + str(cate_id))
        do_inference(prompt)

print("get_random(1)")
print("get_random(2) use this")
print("get_random(3)")
