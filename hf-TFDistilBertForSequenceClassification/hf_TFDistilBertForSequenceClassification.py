
# vars
LOOP_NUM = 10
EPOCHS = 1
DATASET_NUM = 10_000
TRAIN_SET_RATIO = 0.99
BATCH_SIZE = 8 
LEARNING_RATE = 1e-5 # 0.01
TEST_size = 0.01
MAX_SEQ_LEN = 128
num_classes = 2_000

is_first_time_create_model = False
#is_first_time_create_model = True

#pretrained_model_name = 'bert-base-chinese'
#save_model_dir_name = 'bert-base-chinese'
pretrained_model_name = 'distilbert-base-multilingual-cased'
save_model_dir_name = 'distilbert-base-multilingual-cased'

save_model_suffix = '-tf-keras'

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


# load model
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


# load model

model_save_h5_dir = dir_all_model + '/' + save_model_dir_name + save_model_suffix # don't end '.h5'
if not os.path.exists(model_save_h5_dir):
    is_first_time_create_model = True
model, tokenizer = get_model_tokenizer(pretrained_model_name, model_save_h5_dir, num_classes, is_first_time_create_model)
#
import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])



# 
import tensorflow as tf
import pandas as pd
def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,
                                 max_length=MAX_SEQ_LEN,
                                 padding='max_length',
                                 ##return_token_type_ids=True,  ##
                                 return_attention_mask=True)
##def map_example_to_dict(input_ids, token_type_ids, attention_masks, label):
def map_example_to_dict(input_ids, attention_masks, label):
    return {
        "input_ids": input_ids,
        ##"token_type_ids": token_type_ids,
        "attention_mask": attention_masks,
    }, label

def encode_examples(ds, limit=-1):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
    for index, row in ds.iterrows():
        bert_input = convert_example_to_feature(row['sentence'])
        input_ids_list.append(bert_input['input_ids'])
        ##token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append(row['category'])
    ##return tf.data.Dataset.from_tensor_slices((input_ids_list, token_type_ids_list, attention_mask_list, label_list)).map(map_example_to_dict)
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, label_list)).map(map_example_to_dict)


# data
def get_dataset():
    data = {
        'sentence': ['this is sentence A', 'this is sentence B'],
        'category': [1, 2]
    }
    df_data = pd.DataFrame(data)
    df_train_data = pd.concat([df_data] * int(DATASET_NUM / 2 * TRAIN_SET_RATIO), ignore_index=True)
    df_test_data = pd.concat([df_data] * int(DATASET_NUM / 2 * (1 - TRAIN_SET_RATIO)), ignore_index=True)
    _train_data = encode_examples(df_train_data).shuffle(10000).batch(BATCH_SIZE)
    _test_data = encode_examples(df_test_data).batch(BATCH_SIZE)
    return (_train_data, _test_data)


# train the model
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

import tensorflow as tf
class TransformersCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, save_freq=1):
        super(TransformersCheckpointCallback, self).__init__()
        self.save_path = save_path
        self.save_freq = save_freq
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
        #if (epoch + 1) % 10 == 0:
            #save_path = os.path.join(self.save_path, f"epoch_{epoch+1}")
            #os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(self.save_path)
            #print(msg_time(), str(epoch))

print(msg_time(), 'start')
for iii in range(LOOP_NUM):
    # dataset
    train_data, test_data = get_dataset()
    # train, use callback
    ##checkpoint_callback = TransformersCheckpointCallback(model_save_h5_dir)
    ##history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data, callbacks=[checkpoint_callback])
    # train, manually
    history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data)
    print(msg_time(), 'loop:' + str(iii))
    if ((iii + 1) %  2) == 0:
        model.save_pretrained(model_save_h5_dir)
        tokenizer.save_pretrained(model_save_h5_dir)
        print('s' * 80)
    












