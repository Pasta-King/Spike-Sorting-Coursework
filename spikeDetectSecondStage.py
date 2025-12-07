import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from keras import datasets, layers, models, backend, losses
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d


model_version = 1
detector_version = 26

mat = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]

filtered_mat = spio.loadmat("Filtered_Datasets/D1_filtered.mat")
d1_filtered = filtered_mat["re_wave1"]
d1_filtered = d1_filtered[:, 0]

sequence_len = len(d[0])
train_start = 0 
train_end = int(sequence_len * 0.8)

win_size = 25
input_shape = (win_size, 1)
win_step = 25

d_input = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_window = d1_filtered[i:i + win_size]
    d_input.append(d_window)

d_input = np.array(d_input).reshape(-1, win_size)

detector_model = keras.models.load_model("models/spike_detection_v" + str(detector_version) + ".keras")

# detector_model.summary()

output = detector_model.predict(d_input)

x = np.arange(0, sequence_len).tolist()
plt.plot(x, d1_filtered)


relu_flat_output = []
pred_spike_indexes = []
error_spikes = 0

# For Binary
for i in range (0, len(output)):
    if output[i] >= 0.2:
        output_window = [0] * 12 + [1] + [0] * 12
        relu_flat_output = relu_flat_output + output_window

        pred_spike_indexes.append((i * 25) + 12)
    
    else:
        output_window = [0] * 25
        relu_flat_output = relu_flat_output + output_window


sorted_Index = np.sort(Index[0])

spike_pos_sequence = np.zeros(sequence_len, dtype=np.float64)
for i in range(0, len(Index[0])):
    spike_pos_sequence[Index[0][i]]= 1


win_size = 75
input_shape = (win_size, 1)

d_train = []
d_label = []
d_val_train = []
d_val_label = []

for i in range(0, int(len(pred_spike_indexes) * 0.8)):
    d_window = d1_filtered[int(pred_spike_indexes[i]) - 37 : int(pred_spike_indexes[i]) + 38]
    d_train.append(d_window)

    label_window = spike_pos_sequence[int(pred_spike_indexes[i] - 12) : int(pred_spike_indexes[i] + 13)]
    d_label.append(label_window)

for i in range(int(len(pred_spike_indexes) * 0.8), len(pred_spike_indexes)):
    d_window = d1_filtered[int(pred_spike_indexes[i]) - 37 : int(pred_spike_indexes[i]) + 38]
    d_val_train.append(d_window)

    label_window = spike_pos_sequence[int(pred_spike_indexes[i] - 12) : int(pred_spike_indexes[i] + 13)]
    d_val_label.append(label_window)

d_train = np.array(d_train).reshape(-1, win_size)
d_label = np.array(d_label) #.reshape(-1, 200)
d_val_train = np.array(d_val_train).reshape(-1, win_size)
d_val_label = np.array(d_val_label) #.reshape(-1, 200)

model = models.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Normalization(axis=None))
model.add(layers.Conv1D(2, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(4, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(32, 6, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(64, 12, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(128, 24, padding="same", activation="sigmoid"))
model.add(layers.Flatten())
model.add(layers.Dense(1000, activation="sigmoid"))
model.add(layers.Dense(600, activation="sigmoid"))
model.add(layers.Dense(300, activation="sigmoid"))
model.add(layers.Dense(25, activation="sigmoid"))
model.summary()

model.compile(optimizer='adamW', loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])

history = model.fit(d_train, d_label, epochs=240, batch_size=32, validation_data=(d_val_train, d_val_label)) 

model.save("models/spike_detect_second_stage_v" + str(model_version) + ".keras")
