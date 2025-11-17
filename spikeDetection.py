import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
from keras import datasets, layers, models, backend, losses

model_version = 3

mat = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]

sequence_len = len(d[0])

# d_zeroes = [[0, 1]] * sequence_len
# for i in range(0, len(Index[0])):
#     d_zeroes[Index[0][i]] = [1, 0] # Class[0][i]

d_zeroes = np.zeros(sequence_len, dtype=np.float64)
for i in range(0, len(Index[0])):
    d_zeroes[Index[0][i]] = 1 # Class[0][i]

# Output (200, 1) 
# 0 where there isn't a spike and 1 where there is

train_start = 0
train_end = int(sequence_len * 0.8)

win_size = 200
input_shape = (win_size, 1)
win_step = 160

d_train = []
d_label = []
d_val_train = []
d_val_label = []

for i in range(train_start, train_end, win_step):
    d_train.append(d[0][i:i + win_size])
    d_label.append(d_zeroes[i:i + win_size])

for i in range(train_end, sequence_len - win_size, win_step):
    d_val_train.append(d[0][i:i + win_size])
    d_val_label.append(d_zeroes[i:i + win_size])

d_train = np.array(d_train).reshape(-1, win_size)
print(d_train)
d_label = np.array(d_label) #.reshape(-1, 200)
d_val_train = np.array(d_val_train).reshape(-1, win_size)
d_val_label = np.array(d_val_label) #.reshape(-1, 200)


input_shape = (200,1)
model = models.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Conv1D(20, 3, padding="same", activation="sigmoid")) # , input_shape=(200,1)
# model.add(layers.MaxPooling1D(4))
model.add(layers.Conv1D(50, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(50, 10, padding="same", activation="sigmoid"))
# model.add(layers.MaxPooling1D(4))
model.add(layers.Dense(1000, activation="sigmoid"))
model.add(layers.Dense(600, activation="sigmoid"))
# model.add(layers.Dense(200, activation="sigmoid"))
#model.add(layers.Dense(200, activation="sigmoid"))
model.add(layers.Dense(2, activation="sigmoid"))
model.summary()

# def custom_loss(y_true, y_pred):
#     loss = float(0)

#     spike_indexes = np.nonzero(y_true)[0] # np.nonzero creates a tuple of arrays so first element is selected
    

#     if not spike_indexes:
#         loss += backend.log(1 - y_pred)
#     else:
#         midpoints = [0]
#         for i in range(1, len(spike_indexes)):
#             midpoints.append((spike_indexes[i-1] + spike_indexes[i]) // 2)
#         midpoints.append(len(y_true))


    # spike_present = 0
    # for i in range(0, len(y_true)):
    #     if y_true[i] == 1:
    #         spike_present = 1
    #         loss = 0

# if there is a spike, loss = sum (prob at index * distance to true spike index) - prob at true spike index
# if there is no spike, loss = sum (prob at index * mean distance) 

model.compile(optimizer='adamW', loss=losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

history = model.fit(d_train, d_label, epochs=20, batch_size=16, validation_data=(d_val_train, d_val_label)) 

model.save("models/spike_detection_v" + str(model_version) + ".keras")

# output = model.predict(d_train)
# spikes = np.flatnonzero(output > 0.25)
# print(spikes)
# print("Predicted Spikes", len(spikes))
# print(np.sort(Index)[0])
# print("Real Spikes", len(np.sort(Index)[0]))