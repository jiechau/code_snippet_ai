
# https://medium.com/@abhig0303/setting-up-tensorflow-with-cuda-for-gpu-on-windows-11-a157db4dae3e



import sys
print('python: ' + str(sys.version)) # will print out python version
import tensorflow
print('tf: '+ tensorflow.__version__)
import keras
print('keras: ' + keras.__version__)
# gpu
print(tensorflow.test.gpu_device_name())
#
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
