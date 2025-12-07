import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from keras import datasets, layers, models, backend, losses
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

model_version = 18

mat = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]

filtered_mat = spio.loadmat("Filtered_Datasets/D1_Noisy_filtered.mat")
d1_filtered = filtered_mat["re_wave1"]
d1_filtered = d1_filtered[:, 0]

sequence_len = len(d[0])
train_start = 0
train_end = int(sequence_len * 0.8)

win_size = 100
input_shape = (win_size, 1)
win_step = 160

d_input = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_input.append(d1_filtered[i:i + win_size])

d_input = np.array(d_input).reshape(-1, win_size)

train_data = []
label_data = []

num_of_spikes = len(Index[0])

for i in range(-12, 12, 1):
    for x in range(0, num_of_spikes):
        spike_index = Index[0][x]
        if (spike_index + win_size//2 < sequence_len) and (spike_index - win_size//2 >= 0):
            train_data.append(d1_filtered[spike_index + i - win_size//2 : spike_index + i + win_size//2 ])
        elif ((spike_index + win_size//2 < sequence_len)):
            train_data.append(d1_filtered[: spike_index + win_size//2 ])
        elif (spike_index - win_size//2 >= 0):
            train_data.append(d1_filtered[spike_index - win_size//2 :])
        else:
            train_data.append(d1_filtered[:])

        class_options = [0, 0, 0, 0, 0]
        class_options[Class[0][x] - 1] = 1
        label_data.append(class_options)



val_class_train = np.array(train_data[int(num_of_spikes * 0.8):]).reshape(-1, win_size) 
val_class_label = np.array(label_data[int(num_of_spikes * 0.8):])

class_train = np.array(train_data[:int(num_of_spikes * 0.8)]).reshape(-1, win_size)
class_label = np.array(label_data[:int(num_of_spikes * 0.8)])

model = models.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Normalization(axis=None))
model.add(layers.MaxPooling1D(4))
model.add(layers.Conv1D(20, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(50, 6, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(150, 12, padding="same", activation="sigmoid"))
model.add(layers.Flatten())
model.add(layers.Dense(80, activation="sigmoid"))
model.add(layers.Dense(30, activation="sigmoid"))
model.add(layers.Dense(5, activation="sigmoid"))
model.summary()

model.compile(optimizer='adamW', loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])

history = model.fit(class_train, class_label, epochs=120, batch_size=16, validation_data=(val_class_train, val_class_label)) 

model.save("models/spike_classification_v" + str(model_version) + ".keras")
