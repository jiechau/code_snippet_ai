import os
import transformers

MAX_SEQ_LEN = 128
num_classes = 2_000
is_first_time_create_model = False

lt_pretrained_model_name = [
    #'bert-base-chinese',
    'distilbert-base-multilingual-cased',
]

lt_save_model_dir_name = {
    #'bert-base-chinese': 'bert-base-chinese',
    'distilbert-base-multilingual-cased': 'distilbert-base-multilingual-cased',
}
save_model_suffix = '-tf-keras'

colab_dir = '/tmp/tmp/Colab_Notebooks'
HF_HOME = '/tmp/tmp/transformers'
tmp_dir = '/tmp/tmp' # save tmp

# all

# colab_dir
os.environ["colab_dir"] = colab_dir
os.makedirs(colab_dir, exist_ok=True)
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



def get_model_tokenizer(_pretrained_model_name, _model_path, _num_classes, _is_first_time_create_model):
    if _pretrained_model_name == 'bert-base-chinese':
        bb = '''
bert-base-chinese
Model: "tf_bert_for_sequence_classification_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 bert (TFBertMainLayer)      multiple                  102267648
 dropout_209 (Dropout)       multiple                  0
 classifier (Dense)          multiple                  1538000
=================================================================
Total params:     103_805_648 (395.99 MB)
Trainable params: 103_805_648 (395.99 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
None
        '''
        from transformers import BertTokenizer, TFBertForSequenceClassification
        if _is_first_time_create_model:
            model = TFBertForSequenceClassification.from_pretrained(_pretrained_model_name, num_labels=_num_classes)
            tokenizer = BertTokenizer.from_pretrained(_pretrained_model_name)
        else:
            model = TFBertForSequenceClassification.from_pretrained(_model_path)
            tokenizer = BertTokenizer.from_pretrained(_model_path)
        return model, tokenizer

    if _pretrained_model_name == 'distilbert-base-multilingual-cased':
        bb = '''
distilbert-base-multilingual-cased
Model: "tf_distil_bert_for_sequence_classification_1"
________________________________________________________________________
 Layer (type)                        Output Shape              Param #
========================================================================
 distilbert (TFDistilBertMainLayer)  multiple                  134734080
 pre_classifier (Dense)              multiple                  590592
 classifier (Dense)                  multiple                  1538000
 dropout_229 (Dropout)               multiple                  0
========================================================================
Total params: 136862672 (522.09 MB)
Trainable params: 136862672 (522.09 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
None
        '''
        from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
        if _is_first_time_create_model:
            model = TFDistilBertForSequenceClassification.from_pretrained(_pretrained_model_name, num_labels=_num_classes)
            tokenizer = DistilBertTokenizer.from_pretrained(_pretrained_model_name)
        else:
            model = TFDistilBertForSequenceClassification.from_pretrained(_model_path)
            tokenizer = DistilBertTokenizer.from_pretrained(_model_path)
        return model, tokenizer


import tensorflow as tf
import pandas as pd

lt_model = {}
lt_tokenizer = {}
lt_model_save_h5_dir = {}
for pretrained_model_name in lt_pretrained_model_name:
    save_model_dir_name = lt_save_model_dir_name[pretrained_model_name]
    # load model
    model_save_h5_dir = dir_all_model + '/' + save_model_dir_name + save_model_suffix # don't end '.h5'
    lt_model_save_h5_dir['pretrained_model_name'] = model_save_h5_dir
    model, tokenizer = get_model_tokenizer(pretrained_model_name, model_save_h5_dir, num_classes, is_first_time_create_model)
    lt_model[pretrained_model_name] = model
    lt_tokenizer[pretrained_model_name] = tokenizer


# inference
def predict_classification(_model, _tokenizer, _input_sentence):
    bert_input = _tokenizer.encode_plus(_input_sentence,
                                 add_special_tokens=True,
                                 max_length=MAX_SEQ_LEN,
                                 padding='max_length',
                                 return_attention_mask=True,
                                 ##return_token_type_ids=True,
                                 return_tensors='tf')
    input_ids = bert_input['input_ids']
    attention_mask = bert_input['attention_mask']
    ##token_type_ids = bert_input['token_type_ids']
    _predictions = _model.predict({'input_ids': input_ids,
                                 'attention_mask': attention_mask
                                 ##'token_type_ids': token_type_ids
                                 })
    return _predictions

# inference
sentence = 'this is sentence B'


for pretrained_model_name in lt_pretrained_model_name:
    print('')
    print(sentence)
    print(pretrained_model_name)
    model = lt_model[pretrained_model_name]
    tokenizer = lt_tokenizer[pretrained_model_name]
    predictions = predict_classification(model, tokenizer, sentence)
    logits = predictions['logits']
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    print("predicted:", predicted_class)



