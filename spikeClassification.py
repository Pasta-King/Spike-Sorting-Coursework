import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras

model_version = 2

mat = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]

sequence_len = len(d[0])
train_start = 0
train_end = int(sequence_len * 0.8)


d_zeroes = np.zeros(sequence_len, dtype=np.float64)
for i in range(0, len(Index[0])):
    d_zeroes[Index[0][i]] = Class[0][i]

win_size = 200
input_shape = (win_size, 1)
win_step = 160

d_input = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_input.append(d[0][i:i + win_size])

d_input = np.array(d_input).reshape(-1, win_size)

model = keras.models.load_model("models/spike_detection_v" + str(model_version) + ".keras")

output = model.predict(d_input)

class_train = []
class_label = []
relu_flat_output = []
error_spikes = 0

# For SparseCategoricalCrossEntropy
for i in range (0, len(output)):
    for x in range(0, win_step):
        if output[i][x][1] >  output[i][x][0]:
            class_train.append(d[i - 100: i + 100])
            class_label = d_zeroes[i * 160 + x]


