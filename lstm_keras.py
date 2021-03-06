# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 23:04:09 2020

@author: avner
"""


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt

import numpy as np

bin_len = 8 # 8 digits for each int

# list for saving binary repsetantion of int
bin_arr = np.unpackbits(np.array([range(2**bin_len)], dtype=np.uint8).T,axis=1)

max_int = 2**bin_len/2  

int_1 = np.random.randint(max_int, size=(30))

int_2 = np.random.randint(max_int, size=(30))

sum_int = int_1+int_2

X_train = np.concatenate((bin_arr[int_1], bin_arr[int_2]), 1)

y_train = bin_arr[sum_int]


regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()