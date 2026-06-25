# -*- coding: utf-8 -*-
"""
use ppp
"""

# 順序
# hf_llama2_NousResearch.ipynb
# hf_llama2_NousResearch_merge.ipynb
# hf_NousResearch_inference.ipynb


# vars
LOOP_NUM = 1
EPOCHS = 1
DATASET_NUM = 100 #
TRAIN_SET_RATIO = 0.99
TEST_size = 0.01
BATCH_SIZE = 8 # 那台 rog 只能用 8，(distilbert 只能用 4)
LEARNING_RATE = 2e-4 # 1e-5
MAX_SEQ_LEN = 128
num_classes = 2_000
MY_EVAL_STEPS = 10
MY_SAVE_STEPS = 10
MY_SAVE_EPOCHS = 10




# 這個用在  peft
is_first_time_create_model = False
#is_first_time_create_model = True

# loacal hf
is_load_hf_from_local = True
#is_load_hf_from_local = False


# 原先的 llama2
model_name = "NousResearch/Llama-2-7b-chat-hf"
# 算好的 peft
save_model_dir_name = 'nousresearch-llama-2-7b-chat-hf'
save_model_suffix = '-pt-qlora-peft'
# 最後合併好之後，要存的
custom_model_dir_name= 'nousresearch-llama-2-7b-custom'


import os
colab_dir = '/tmp/tmp/Colab_Notebooks'
HF_HOME = '/tmp/tmp/transformers'
tmp_dir = '/tmp/tmp' # save tmp

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


# 第一次才需要這個,
# 連 base model 都沒有的時候

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
        # ref
        ns_this, model_this = _model_name.split('/')
        hub_ref = '/hub/models--' + ns_this + '--' + model_this
        # saved
        sub_dir = '/' + ns_this + '/' + model_this
    else:
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



# 最終要的是 df['prompt'],df['response']
import pandas as pd
from datasets import load_dataset
def get_dataset():



    # 最終要的是 df['prompt'],df['response']
    # the dummy data
    data = {
        'prompt': ['what is 2 + 2', 'explain the solar system'],
        'response': ['the answer is 4', 'it is a system that bahla bahla']
    }
    df_data = pd.DataFrame(data)
    df = pd.concat([df_data] * 100, ignore_index=True)




    # 最終要的是 df['prompt'],df['response']
    # 看一下內容
    print(df.head(3))
    # 拆分訓練資料及測試資料
    train_df = df.sample(frac=TRAIN_SET_RATIO, random_state=42) # 這會每次都一樣
    #train_df = df.sample(frac=TRAIN_SET_RATIO)
    test_df = df.drop(train_df.index)
    print(train_df.head(3))

    # 存成jsonl檔
    train_df.to_json(tmp_dir + '/train.jsonl', orient='records', lines=True)
    test_df.to_json(tmp_dir + '/test.jsonl', orient='records', lines=True)

    # 讀取資料集
    train_dataset_json = load_dataset('json', data_files=tmp_dir + '/train.jsonl', split="train")  # 從JSON文件中載入訓練數據集
    valid_dataset_json = load_dataset('json', data_files=tmp_dir + '/test.jsonl', split="train")  # 從JSON文件中載入驗證數據集

    # 對數據集進行前處理，將提示和回應組合成文本對
    _train_dataset = train_dataset_json.map(lambda examples: {'text': [prompt + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)
    _valid_dataset = valid_dataset_json.map(lambda examples: {'text': [prompt + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)

    # return
    return _train_dataset,_valid_dataset

#train_dataset, valid_dataset = get_dataset()
train_data, test_data = get_dataset()



# 匯入必要的模組和套件
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

################################################################################
# Quantized LLMs with Low-Rank Adapters (QLoRA) parameters
################################################################################
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters 輕量級封裝，專門用於CUDA自定義函數，特別是8位優化器、矩陣乘法和量化
################################################################################
use_4bit = True
bnb_4bit_compute_dtype = "float16" # float16 or bfloat16
bnb_4bit_quant_type = "nf4" # fp4 or nf4
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################
output_dir = tmp_dir + "/results"
os.makedirs(output_dir, exist_ok=True)

num_train_epochs = EPOCHS
fp16 = False
bf16 = False
if True:
    bf16 = True
per_device_train_batch_size = BATCH_SIZE #4
per_device_eval_batch_size = BATCH_SIZE # 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = LEARNING_RATE
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 1_000_000# 10
logging_steps = 1_000_000 #10

################################################################################
# Supervised finetuning (SFT) parameters
################################################################################
#max_seq_length = None
max_seq_length = 256 # max 1024
packing = False
device_map = {"": 0} #{"": 0} or "auto"



bb = '''
# 原先的 llama2
model_name = "NousResearch/Llama-2-7b-chat-hf"
# 算好的 peft
save_model_dir_name = 'nousresearch-llama-2-7b-chat-hf'
save_model_suffix = '-pt-qlora-peft'
# 最後合併好之後，要存的
custom_model_dir_name= 'nousresearch-llama-2-7b-custom'
'''

# 算好的 peft
new_model = dir_all_model + '/' + save_model_dir_name + save_model_suffix # don't end '.h5'
if not os.path.exists(new_model):
    is_first_time_create_model = True
os.environ["new_model"] = new_model
os.makedirs(new_model, exist_ok=True)

# 最後合併好之後，要存的
saved_combined_model = dir_all_model + '/' + custom_model_dir_name # don't end '.h5'
os.environ["saved_combined_model"] = saved_combined_model
os.makedirs(saved_combined_model, exist_ok=True)


# 定義位元和字節量化的相關配置
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# 檢查 GPU 是否與 bfloat16 相容
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)


# 從預訓練模型中載入自動生成模型
if is_load_hf_from_local == True:
    load_location = load_hf_pretrained(model_name)
else:
    load_location = model_name
# a check
if not os.path.isdir(load_location):
    load_location = model_name
# 模型
model = AutoModelForCausalLM.from_pretrained(
    load_location,
    quantization_config=bnb_config, # here
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1
# 載入與模型對應的分詞器
tokenizer = AutoTokenizer.from_pretrained(load_location, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# 定義 Prompt Engineering Fine-Tuning （PEFT）的相關設定
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# 設置訓練參數
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    # report_to="tensorboard", #"all"
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=MY_EVAL_STEPS #5  # 每5步驗證
)

from transformers import TrainerCallback
class StepCheckpointCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        pass
        current_step = state.global_step
        if current_step % MY_SAVE_STEPS == 0:
            # Do something every 100 steps
            #print("ss" * 40)
            #print(f"Step {current_step} reached!")
            trainer.model.save_pretrained(new_model, safe_serialization=True) # directory
            #print("saved")
            # ... additional actions
        aa = '''
    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = state.epoch
        num_train_epochs = state.num_train_epochs
        if current_epoch % MY_SAVE_EPOCHS == 0:
            # Do something every 500 steps
            print("ee" * 40)
            trainer.model.save_pretrained(new_model, safe_serialization=True) # directory
            print(f"saved!!!!!! current epoch: {current_epoch} / {num_train_epochs}")
            # ... additional actions
        '''



train_data, test_data = get_dataset()
# 使用 SFTTrainer 進行監督式微調訓練
trainer = SFTTrainer(
    # callbacks=[StepCheckpointCallback()],
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length, # 256
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)


#
bb = '''
def from_pretrained(cls,
model: PreTrainedModel,
model_id: Union[str, os.PathLike],
adapter_name: str='default',
is_trainable: bool=False,
config: Optional[PeftConfig]=None,
**kwargs: Any)
'''
if not is_first_time_create_model:
    trainer.model.from_pretrained(
        model=model,
        model_id = new_model, # lora
#        train_dataset=train_data,
#        eval_dataset=test_data,
#        peft_config=peft_config,
#        dataset_text_field="text",
#        max_seq_length=max_seq_length, # 256
#        tokenizer=tokenizer,
#        args=training_arguments,
#        packing=packing,
    )
    print ('previous petf loaded!')

# 開始訓練模型
# Colab T4: 系統 RAM 3.5 / 12.7 GB, GPU RAM 5.9 / 15.0 GB
for iii in range(LOOP_NUM):
    # 使用 SFTTrainer 進行監督式微調訓練
    trainer = SFTTrainer(
        callbacks=[StepCheckpointCallback()],
        model=model,
        train_dataset=train_data, 
        eval_dataset=test_data,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    # train
    trainer.train()

# 儲存微調後的模型
#trainer.model.save_pretrained(new_model) # directory






#
# 以下是單次測試
#

# trainer.train()

bb = '''
def save_pretrained(
    save_directory: str,
    safe_serialization: bool=False,
    selected_adapters: Optional[List[str]]=None,
    **kwargs: Any)
This function saves the adapter model and the adapter configuration files to a directory, so that it can be
reloaded using the [LoraModel.from_pretrained] class method, and also used by the [LoraModel.push_to_hub]
method.
'''
# trainer.model.save_pretrained(new_model, safe_serialization=True) # directory

# ! ls -l $new_model



#del trainer
ccc = '''
del model
import gc # 清理垃圾桶
gc.collect()
gc.collect()
import time
time.sleep(60)
'''

'''
# Colab T4: 系統 RAM 3.5 / 12.7 GB, GPU RAM 5.9 / 15.0 GB
trainer.train()
# 儲存微調後的模型
trainer.model.save_pretrained(new_model) # directory
'''





eee = '''
/usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:381: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
/usr/local/lib/python3.10/dist-packages/peft/utils/other.py:102: FutureWarning: prepare_model_for_int8_training is deprecated and will be removed in a future version. Use prepare_model_for_kbit_training instead.
/usr/local/lib/python3.10/dist-packages/trl/trainer/sft_trainer.py:159: UserWarning: You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to 1024
You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. It is recommended to update your version as a major bug has been fixed in 8-bit optimizers.
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
/usr/local/lib/python3.10/dist-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.

'''

# DATASET_NUM = -1 # 610_012
# Map: 100% 549011/549011 [01:12<00:00, 5503.89 examples/s]
# Map: 100% 61001/61001 [00:08<00:00, 8850.68 examples/s]
# [ 6/137253 00:07 < 68:05:46, 0.56 it/s, Epoch 0.00/1]
# [ 249/7626 04:17 < 2:07:51, 0.96 it/s]

# DATASET_NUM = 1_000
# Map: 100% 900/900 [00:00<00:00, 3360.36 examples/s]
# Map: 100% 100/100 [00:00<00:00, 1831.86 examples/s]
# [ 76/225 04:47 < 09:38, 0.26 it/s, Epoch 0.33/1] [225/225 16:04, Epoch 1/1]
# [ 7/13 00:05 < 00:05, 1.12 it/s]


# %load_ext tensorboard
# %tensorboard --logdir results/runs

# 日誌輸出
# logging.set_verbosity(logging.CRITICAL)

# 執行模型的文本生成流程
ddd = '''
#prompt = "What is a large language model?"
q = "'【funcare 船井生醫】高單位葉黃素(30顆)x3盒'"
#q = "北歐風大容量304不鏽鋼泡麵碗1000ML"
prompt = "you are a 'special category id' toolbox, when i give you a sentence, you give me its 'special category id'. now, tell me what is the 'special category id' of this sentence: " + q
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text']) # 輸出生成的文本
'''

# result

# Empty VRAM
aaa = '''
# Empty VRAM
del model
del pipe
del trainer
import gc # 清理垃圾桶
gc.collect()
gc.collect()
'''


