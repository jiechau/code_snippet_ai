
#import os
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras

# load model
model = keras.models.load_model('/tmp/tmp/my_model_tf') 

# random pick one
import numpy as np
from PIL import Image
(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
index = np.random.randint(0, len(x_train))
img = x_train[index].reshape(28,28)
img = Image.fromarray(img) 

img_array = keras.preprocessing.image.img_to_array(img)
img_array = img_array.reshape(1, 28, 28, 1) / 255.0

import time
start = time.time()
# inference
predictions = model.predict(img_array)
end = time.time()
print('')
print(end - start)

# 得到预测结果类别
predicted_index = np.argmax(predictions[0])
predicted_label = str(predicted_index)

print("Predicted digit: ", predicted_label)
#print("predictions: ", predictions)
for row_index, row in enumerate(predictions):
    for col_index, value in enumerate(row):
        print(f"{col_index} - {value}")
