import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
from keras import datasets, layers, models, backend, losses
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

model_version = 31

mat = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]

filtered_mat = spio.loadmat("Filtered_Datasets/D1_Noisy_filtered.mat")
d1_filtered = filtered_mat["re_wave1"]
d1_filtered = d1_filtered[:, 0]

sequence_len = len(d[0])

d_zeroes = np.zeros(sequence_len, dtype=np.float64)
for i in range(0, len(Index[0])):
    d_zeroes[Index[0][i]]= 1


train_start = 0
train_end = int(sequence_len * 0.8)

win_size = 50
input_shape = (win_size, 1)
win_step = 30

d_train = []
d_label = []
d_val_train = []
d_val_label = []

for i in range(train_start, train_end, win_step):
    d_window = d1_filtered[i:i + win_size]
    d_train.append(d_window)

    if np.any(d_zeroes[i: i + win_size]):
        d_label.append([1])
    else:
        d_label.append([0])

for i in range(train_end, sequence_len - win_size, win_step):
    d_window = d1_filtered[i:i + win_size]
    d_val_train.append(d_window)

    if np.any(d_zeroes[i: i + win_size]):
        d_val_label.append([1])
    else:
        d_val_label.append([0])

d_train = np.array(d_train).reshape(-1, win_size)
d_label = np.array(d_label) #.reshape(-1, 200)
d_val_train = np.array(d_val_train).reshape(-1, win_size)
d_val_label = np.array(d_val_label) #.reshape(-1, 200)


model = models.Sequential()
model.add(layers.Input(shape=input_shape))
# model.add(layers.Normalization(axis=None))
model.add(layers.Conv1D(16, 3, padding="same", activation="relu")) # , input_shape=(200,1)
model.add(layers.Conv1D(32, 3, padding="same", activation="relu"))
model.add(layers.Conv1D(64, 3, padding="same", activation="relu"))
model.add(layers.Conv1D(128, 3, padding="same", activation="relu"))
model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Flatten())
model.add(layers.Dense(64, activation="sigmoid"))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()


model.compile(optimizer='adamW', loss=losses.BinaryCrossentropy(), metrics=["accuracy"])

history = model.fit(d_train, d_label, epochs=20, batch_size=18, validation_data=(d_val_train, d_val_label)) 

model.save("models/spike_detection_v" + str(model_version) + ".keras")




