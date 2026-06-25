# 順序
# friDay_cate_id_hugging_face_llama2_NousResearch.ipynb
# friDay_cate_id_hugging_face_llama2_NousResearch_merge.ipynb
# friDay_cate_id_hugging_face_llama2_NousResearch_inference.ipynb

is_colab = False
is_rog = False # that C:/ problem， and gram problem

is_first_time_create_model = False # 這個可以
#is_first_time_create_model = True # 這個可以

is_load_hf_from_local = True

TRAIN_SET_RATIO = 0.99
EPOCHS = 1
DATASET_NUM = 100 #
#DATASET_NUM = -1 # bigdata
MAX_SEQ_LEN = 128
num_classes = 2_000


# 原先的 llama2 以FP16重新載入模型
model_name = "NousResearch/Llama-2-7b-chat-hf"
# 算好的 peft
save_model_dir_name = 'nousresearch-llama-2-7b-chat-hf'
save_model_suffix = '-pt-qlora-peft'
# 最後合併好之後，要存的
custom_model_dir_name= 'nousresearch-llama-2-7b-custom'



# 這裡要注意 用 .py 以下都要 mark
ccc = '''
# 安裝所需的 Python 套件
if is_colab:
    #!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers trl==0.4.7
    #!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.41.3.post2 transformers==4.31.0 trl==0.4.7 tensorboard==2.15.1
    !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.41.3.post2 trl==0.4.7 tensorboard==2.15.1

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
'''


# !pip freeze | grep transformers # transformers==4.31.0





aaa = '''
# Empty VRAM
del model
del pipe
del trainer
import gc # 清理垃圾桶
gc.collect()
gc.collect()
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

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, PeftModel
device_map = {"": 0} #{"": 0} or "auto"

# 算好的 peft
new_model = dir_all_model + '/_saved_model_friDay/friDay_category_id' + '/' + save_model_dir_name + save_model_suffix # don't end '.h5'
os.environ["new_model"] = new_model
os.makedirs(new_model, exist_ok=True)
# 最後合併好之後，要存的
saved_combined_model = dir_all_model + '/_saved_model_friDay/friDay_category_id' + '/' + custom_model_dir_name # don't end '.h5'
os.environ["saved_combined_model"] = saved_combined_model
os.makedirs(saved_combined_model, exist_ok=True)

#
model_save_h5_dir = saved_combined_model
os.environ["model_save_h5_dir"] = model_save_h5_dir
os.makedirs(model_save_h5_dir, exist_ok=True)
#print('------------------------------------------------------------------')
#print(model_save_h5_dir)
#print('------------------------------------------------------------------')

# 原先的 llama2 以FP16重新載入模型
# 12G
if is_load_hf_from_local == True:
    load_location = load_hf_pretrained(model_name)
else:
    load_location = model_name
# a check
if not os.path.isdir(load_location):
    load_location = model_name
print('base_model:', load_location)
# 以FP16重新載入模型並將其與LoRA權重合併
aa = msg_time()
print(msg_time(), 'start load base_model')
base_model = AutoModelForCausalLM.from_pretrained(
    load_location,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    #device_map=device_map,
)
if not is_colab:
    print(msg_time(), 'done load base_model')
    base_model.to("cpu")
    print(msg_time(), 'done .to("cpu") base_model')
# 重新載入分詞器以進行保存
tokenizer = AutoTokenizer.from_pretrained(load_location, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(msg_time(), 'done load tokenizer')
print(' ')


# 算好的
#Colab T4
# System RAM 5.3 / 12.7 GB
# GPU RAM 13.1 / 15.0 GB
print('new_model:', new_model)
aa = msg_time()
print(msg_time(), 'start petf load')
model = PeftModel.from_pretrained(base_model, new_model)
print(msg_time(), 'done petf load')
model = model.merge_and_unload()
print(msg_time(), 'done petf merge')
print(' ')

# 最後要存的
# 儲存合併後的模型
# save
print('model_save_h5_dir:', model_save_h5_dir)
aa = msg_time()
print(msg_time(), 'start save model')
model.save_pretrained(model_save_h5_dir, safe_serialization=True)
tokenizer.save_pretrained(model_save_h5_dir)
print(msg_time(), 'done save model')


ccc = '''
/usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:381: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:386: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:527: UserWarning: The generation config instance is invalid -- `.validate()` throws warnings and/or exceptions. Fix these issues to save the configuration. This warning will be raised to an exception in v4.34.

`do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.

('/content/drive/MyDrive/Colab_Notebooks/_saved_model/_saved_model_friDay/friDay_category_id/nousresearch-llama-2-7b-chat-hf-pt-qlora-peft/tokenizer_config.json',
 '/content/drive/MyDrive/Colab_Notebooks/_saved_model/_saved_model_friDay/friDay_category_id/nousresearch-llama-2-7b-chat-hf-pt-qlora-peft/special_tokens_map.json',
 '/content/drive/MyDrive/Colab_Notebooks/_saved_model/_saved_model_friDay/friDay_category_id/nousresearch-llama-2-7b-chat-hf-pt-qlora-peft/tokenizer.json')

 '''

#! ls -l $model_save_h5_dir

#!ls -l /content/drive/MyDrive/Colab_Notebooks/_saved_model/_saved_model_friDay/friDay_category_id/nousresearch-llama-2-7b-chat-hf-pt-qlora-peft
