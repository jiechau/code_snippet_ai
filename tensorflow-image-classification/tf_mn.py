
# https://medium.com/@abhig0303/setting-up-tensorflow-with-cuda-for-gpu-on-windows-11-a157db4dae3e
# Python — — — — — — — — 3.8 (3.9 ok too)
# Tensorflow — — — — — —2.5 (only 2.5 works)
# Keras — — — — — — — — — 2.5
# CUDA Toolkit — — — — — 11.8.0
# cuDNN library — — — — — 8.6.0

# conda create -n py39tf25 python=3.9; conda activate py39tf25
# # only pip install works (not conda install)
# pip install tensorflow==2.5

import tensorflow as tf
from tensorflow import keras

import json
import os

#tf_config = json.loads(os.environ['TF_CONFIG'])
#cluster_spec = tf_config['cluster'] 
#task_type = tf_config['task']['type']
#task_id = tf_config['task']['index']
#num_workers = len(tf_config['cluster']['worker'])
#print(tf_config)


def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# ndarray, 60000 train, and 10000 test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# dataset
num_workers = 2
per_worker_batch_size = 64
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(global_batch_size)


# (1) build
model = build_model()
# (2) load
#model = keras.models.load_model('C:\share\tmp\my_model_mn') # all workers should use chief's version

# compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#callbacks = [tf.keras.callbacks.ModelCheckpoint('C:/tmp/my_model_tf', save_freq='epoch')]
callbacks = [tf.keras.callbacks.ModelCheckpoint('/tmp/tmp/my_model_tf', save_freq='epoch')]

# experimental_distribute_dataset
#dist_dataset = strategy.experimental_distribute_dataset(multi_worker_dataset) 

import time
start = time.time()
#model.fit(x_train, y_train, epochs=2, batch_size=64) # default batch_size=32
model.fit(multi_worker_dataset, epochs=2, steps_per_epoch=int(60000/global_batch_size))
#model.fit(multi_worker_dataset, epochs=2, steps_per_epoch=int(60000/global_batch_size), callbacks=callbacks)
#model.fit(dist_dataset, epochs=10, steps_per_epoch=int(60000/global_batch_size), callbacks=callbacks)
end = time.time()
print(end - start)
print('')

# evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print('global_batch_size', global_batch_size)

# this way, every worker save model
# but not a good practice
#model.save('/tmp/tmp/my_model_tf')
