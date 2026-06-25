# check_sys_info
# some ai template

git clone https://gitlab.com/jiechau/check_sys_info.git

## uv

## BAAI's BGE-M3 
# with FlagEmbedding (its dev tool)
# inline uv (no need pyproject.toml uv.lock)
uv run bge_m3.py

## venv
## venv: py39 tf39 tf39cpu pt39 pt311
# tf39
/usr/local/bin/python3.9 -m venv --system-site-packages tf39
source tf39/bin/activate
pip install -r requirements.tf39.pip.txt
# tf39cpu
/usr/local/bin/python3.9 -m venv --system-site-packages tf39cpu
source tf39cpu/bin/activate
pip install -r requirements.tf39cpu.pip.txt
# pt39 (used in: gpu or cpu)
/usr/local/bin/python3.9 -m venv --system-site-packages pt39
source pt39/bin/activate
pip install -r requirements.pt39.pip.txt
# pt311 (used in: gpu or cpu)
/usr/local/bin/python3.11 -m venv --system-site-packages pt311
source pt311/bin/activate
pip install -r requirements.pt311.pip.txt


## tensorflow regression
# use venv tf39 tf39cpu to switch gpu/cpu
python tf_dnn.py

## torch regression
# use venv pt39 
pt_dnn.py


## tensorflow image classification
# use venv tf39 tf39cpu to switch gpu/cpu
python tf_mn.py
python tf_mni.py (just inference. need to unmark last line of pt_mn.py)

## torch image classification
# use venv pt39 
pt_mn.py
pt_mni.py (just inference. need to unmark last line of pt_mn.py)
pt_mn_cpu.py
pt_mni_cpu.py (just inference. need to unmark last line of pt_mn_cpu.py)


## sentencetransformer
# pt_sentencetransformer.requirements.cuda.pip.txt
# pt_sentencetransformer.requirements.cpu.pip.txt
/usr/local/bin/python3.9 -m venv --system-site-packages sss
source sss/bin/activate
pip install --upgrade pip
pip install -r pt_sentencetransformer.requirements.cuda.pip.txt
# or cpu version: pip install -r pt_sentencetransformer.requirements.cpu.pip.txt
python pt_sentencetransformer1.py
python pt_sentencetransformer2.py
python pt_sentencetransformer3.py


## spacy
# requirements.py39spacy.pip.txt
/usr/local/bin/python3.9 -m venv --system-site-packages py39spacy
source py39spacy/bin/activate
pip install --upgrade pip
pip install spacy
pip install spacy[transformers]
pip install cupy-cuda12x # if GPU, see nvidia-smi or nvcc --version
python -m spacy download zh_core_web_trf
python -m spacy info
# inference
python -W ignore spacy_inference.py
# train data
python spacy_data.py
# https://spacy.io/usage/training#config
# zh, ner, and save to base_config.cfg
python -m spacy init fill-config ./base_config.cfg ./config.cfg
python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy --gpu-id 0 # --training.max_epochs 10
# inference
python -W ignore spacy_inference_new.py



## hugging face TFDistilBertForSequenceClassification
# use venv hf39tf 
/usr/local/bin/python3.9 -m venv --system-site-packages hf39tf
source hf39tf/bin/activate
pip install --upgrade pip
pip install -r requirements.hf39tf.pip.txt
python hf_TFDistilBertForSequenceClassification.py
python hf_TFDistilBertForSequenceClassification_inference.py

## hugging face TFAutoModelForImageClassification
# use venv hf39tf 
/usr/local/bin/python3.9 -m venv --system-site-packages hf39tf
source hf39tf/bin/activate
pip install --upgrade pip
pip install -r requirements.hf39tf.pip.txt
python hf_TFAutoModelForImageClassification.py
python hf_TFAutoModelForImageClassification_inference.py

## hugging face AutoModelForImageClassification
# use venv hf39pt
/usr/local/bin/python3.9 -m venv --system-site-packages hf39pt
source hf39pt/bin/activate
pip install --upgrade pip
pip install -r requirements.hf39pt.pip.txt
python hf_AutoModelForImageClassification.py
python hf_AutoModelForImageClassification_inference.py

## hugging face llama2 NousResearch
# use venv ppp 
/usr/local/bin/python3.9 -m venv --system-site-packages ppp
source ppp/bin/activate
pip install --upgrade pip
pip install -r requirements.ppp.pip.txt
# split to 3 part, in case of lack of resource (ram/vram)
hf_llama2_NousResearch.py # train
hf_llama2_NousResearch_merge.py # merge after train. save to custom model
hf_llama2_NousResearch_inference.py # use custom model to do inference

## microsoft graphRAG
# this one is just note, in 2024/07 openai launch gpt-4o-mini which is very 'cheap' to make things possible
everything follow the tutorial: https://microsoft.github.io/graphrag/posts/get_started/
activate your openai 'Organization account' (not monthly user subscription) by deposit $10 https://platform.openai.com/settings/organization/billing/overview
create api key https://platform.openai.com/organization/api-keys
create a virtual env, e.g. py312graphrag
you'll need access to openai's chat and embeddings api
edit ragtest/.env put api key
edit ragtest/settings.yaml # model: gpt-4o-mini
get txt file Alice's Adventures in Wonderland https://manybooks.net/titles/carrolll2888528885-8.html
put txt file in ragtest/input
calculate token https://platform.openai.com/tokenizer
# example
check_sys_info/graphrag_book/ragtest_az/settings.yaml
check_sys_info/graphrag_book/ragtest_openai/settings.yaml
# read
check_sys_info/graphrag_book/README.txt


## docker
# tensorflow / GPU
docker run -it --rm --gpus all tensorflow/tensorflow:latest-gpu bash
# tensorflow / CPU
docker run -it --rm tensorflow/tensorflow:latest bash # "Ubuntu 22.04.3 LTS (Jammy Jellyfish)"
# inside container bash, run these:
nvidia-smi
apt update; apt install git
git clone https://gitlab.com/jiechau/check_sys_info.git
cd check_sys_info
python tf_mn.py # every epoch should be less than 5 sec

## docker
# pytorch / gpu or cpu
docker run -it --rm --gpus all nvcr.io/nvidia/pytorch:23.10-py3 bash
# inside container bash, run these:
git clone https://gitlab.com/jiechau/check_sys_info.git
nvidia-smi
cd check_sys_info
python pt_mn.py # every epoch should be less than 10 sec
python pt_mn_cpu.py



P.S.

## nvidia driver on ubuntu 22.04
https://ivonblog.com/posts/ubuntu-install-nvidia-drivers/
  apt install nvidia-driver-545
  GeForce GTX 1060 3GB (2016)
  VGA compatible controller: NVIDIA Corporation GP106 [GeForce GTX 1060 3GB] (rev a1)
  GeForce RTX 3060 12G (2021)
  VGA compatible controller: NVIDIA Corporation GA106 [GeForce RTX 3060 Lite Hash Rate] (rev a1)

## nvidia driver on ROG Flow X16 (2022) GV601 GV601RM-0042E6900HS
## RTX 3060
https://medium.com/@abhig0303/setting-up-tensorflow-with-cuda-for-gpu-on-windows-11-a157db4dae3e
ROG Flow X16 (2022) GV601 GV601RM-0042E6900HS
  1TB PCIe® 4.0 NVMe™ M.2 Performance SSD
  8GB DDR5-4800 SO-DIMM x 2
  AMD Ryzen™ 9 6900HS Mobile Processor (8-core/16-thread, 16MB cache, up to 4.9 GHz max boost)
  NVIDIA® GeForce RTX™ 3060 Laptop GPU ROG Boost: 1475MHz* at 125W (1425MHz Boost Clock+50MHz OC, 100W+25W Dynamic Boost) 6GB GDDR6
  RTX 3060
Operating System : Windows 11 Home
Graphics Card: NVIDIA GPU RTX-3060
# Python — — — — — — — — 3.8 (3.9 ok too)
# Tensorflow — — — — — —2.5 (only 2.5 works)
# Keras — — — — — — — — — 2.5
# CUDA Toolkit — — — — — 11.8.0
# cuDNN library — — — — — 8.6.0

# conda create -n py39tf25 python=3.9; conda activate py39tf25
# # only pip install works (not conda install)
# pip install tensorflow==2.5

go_conda.bat
%windir%\System32\cmd.exe "/K" C:\ProgramData\anaconda3\Scripts\activate.bat C:\ProgramData\anaconda3

cdj.bat
cd C:\share\jiechau\ml_codes

conda env list
conda activate py39tf25
conda activate py39tf25cpu
conda activate py39pt210pip
conda activate py39pt210conda
conda activate py39pt210cpu
conda activate ttt






