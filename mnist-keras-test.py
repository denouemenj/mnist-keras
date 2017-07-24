import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.optimizers import Adam
from PIL import Image
from keras import backend as K

from keras.models import load_model

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.
x_test = x_test.reshape(x_test.shape[0], -1) / 255.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print(K.image_data_format())
model=load_model('mnist-keras.h5')

fname = '8.png'
image = Image.open(fname).convert("L")
image = image.resize((28,28),Image.ANTIALIAS)
arr = np.asarray(image)
print(arr.shape)
arr=1-arr/255
arr=np.reshape(arr,(1,28,28,1))
print(arr.shape)

out1 = model.predict_classes(arr)
print("print after load:", out1)