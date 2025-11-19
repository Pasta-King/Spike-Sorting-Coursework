import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from keras import datasets, layers, models, backend, losses

model_version = 5

mat = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]

sequence_len = len(d[0])
train_start = 0
train_end = int(sequence_len * 0.8)

win_size = 100
input_shape = (win_size, 1)
win_step = 160

d_input = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_input.append(d[0][i:i + win_size])

d_input = np.array(d_input).reshape(-1, win_size)

# detection_model = keras.models.load_model("models/spike_detection_v" + str(model_version) + ".keras")

# output = detection_model.predict(d_input)

train_data = []
label_data = []

num_of_spikes = len(Index[0])

for i in range(0, num_of_spikes):
    spike_index = Index[0][i]
    if (spike_index + win_size//2 < sequence_len) and (spike_index - win_size//2 >= 0):
        train_data.append(d[0][spike_index - win_size//2 : spike_index + win_size//2 ])
    elif ((spike_index + win_size//2 < sequence_len)):
        train_data.append(d[0][: spike_index + win_size//2 ])
    elif (spike_index - win_size//2 >= 0):
        train_data.append(d[0][spike_index - win_size//2 :])
    else:
        train_data.append(d[0][:])

    class_options = [0, 0, 0, 0, 0]
    class_options[Class[0][i] - 1] = 1
    label_data.append(class_options)

val_class_train = np.array(train_data[int(num_of_spikes * 0.8):]).reshape(-1, win_size) 
val_class_label = np.array(label_data[int(num_of_spikes * 0.8):])

class_train = np.array(train_data[:int(num_of_spikes * 0.8)]).reshape(-1, win_size)
class_label = np.array(label_data[:int(num_of_spikes * 0.8)])

model = models.Sequential()
model.add(layers.Input(shape=input_shape))
#model.add(layers.Conv1D(30, 3, padding="same", activation="sigmoid"))
#model.add(layers.MaxPooling1D(4))
model.add(layers.Conv1D(20, 3, padding="same", activation="relu"))
model.add(layers.MaxPooling1D(4))
model.add(layers.Conv1D(30, 3, padding="same", activation="relu"))
#model.add(layers.MaxPooling1D(4))
# model.add(layers.Conv1D(50, 10, padding="same", activation="relu"))
# model.add(layers.MaxPooling1D(4))
model.add(layers.Flatten())
# model.add(layers.Dense(600, activation="relu"))
# model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(20, activation="relu"))
model.add(layers.Dense(30, activation="relu"))
model.add(layers.Dense(5, activation="sigmoid"))
model.summary()

model.compile(optimizer='adamW', loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])

# print(class_train)

history = model.fit(class_train, class_label, epochs=60, batch_size=16, validation_data=(val_class_train, val_class_label)) 

model.save("models/spike_classification_v" + str(model_version) + ".keras")

# # For SparseCategoricalCrossEntropy
# for i in range (0, len(output)):
#     for x in range(0, win_step):
#         if output[i][x][1] >  output[i][x][0]:
#             class_train.append(d[i - 100: i + 100])
#             class_label = d_zeroes[i * 160 + x]


