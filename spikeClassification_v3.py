import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from keras import datasets, layers, models, backend, losses
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

model_version = 23
detector_version = 32

mat = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]

filtered_mat = spio.loadmat("Filtered_Datasets/D1_Noisy_filtered.mat")
d1_filtered = filtered_mat["re_wave1"]
d1_filtered = d1_filtered[:, 0]

sequence_len = len(d[0])
train_start = 0 

win_size = 50
input_shape = (win_size, 1)
win_step = 30

d_input = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_window = d1_filtered[i:i + win_size]
    d_input.append(d_window)

d_input = np.array(d_input).reshape(-1, win_size)

second_detect_model = keras.models.load_model("models/spike_detection_v" + str(detector_version) + ".keras")


print("Second Stage")
output = second_detect_model.predict(d_input)# [0, :, 0]
    
relu_flat_output = []
for i in range (0, len(output)):
    for x in range(0, win_step):
        if output[i][x] >  0.5:
            prob_window = np.append(0, output[i][x-10:x+10])
            max_index = np.argmax(prob_window)
            if max_index == 11:
                relu_flat_output.append(1)
            else:
                relu_flat_output.append(0)
        else:
            relu_flat_output.append(0)


precise_pred_indexes = np.nonzero(relu_flat_output)[0]

train_start = 0

upper_size = 80

win_size = 200
input_shape = (win_size, 1)

class_pos_sequence = np.zeros(sequence_len, dtype=np.float64)
for i in range(0, len(Index[0])):
    class_pos_sequence[Index[0][i]]= Class[0][i]

d_train = []
d_label = []
d_val_train = []
d_val_label = []

for i in range(0, int(len(precise_pred_indexes) * 0.8)):
    for x in range(-6, 6):
        
        if (precise_pred_indexes[i] + x - (win_size//2)) < 0:
            d_window = d1_filtered[0 : int(precise_pred_indexes[i] + (win_size//2) + x)]
            upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : int(precise_pred_indexes[i] + (win_size//2) + x)]
            bottom_half = class_pos_sequence[0: int(precise_pred_indexes[i] + x)]
            avg = np.mean(d_window)
            padding_size = int((win_size//2) - len(bottom_half))
            d_window = np.concatenate([[avg] * padding_size, d_window], axis=0) # * int((win_size//2) - precise_pred_indexes[i])
            bottom_half = np.concatenate([[0] * padding_size, bottom_half], axis=0) 
        elif (precise_pred_indexes[i] + x + (win_size//2)) > len(d1_filtered):
            d_window = d1_filtered[int(precise_pred_indexes[i] - (win_size//2)) : ]
            upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : ]
            bottom_half = class_pos_sequence[int(precise_pred_indexes[i] - (win_size//2) + x) : int(precise_pred_indexes[i] + x)]
            avg = np.mean(d_window)
            padding_size = int((win_size//2) - len(upper_half))
            d_window = np.concatenate([d_window, [avg] * padding_size ], axis=0)
            upper_half = np.concatenate([upper_half, [0] * padding_size], axis=0)
        else:
            d_window = d1_filtered[int(precise_pred_indexes[i] + x - (win_size//2)) : int(precise_pred_indexes[i] + x + (win_size//2))]
            upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : int(precise_pred_indexes[i] + (win_size//2) + x)]
            bottom_half = class_pos_sequence[int(precise_pred_indexes[i] - (win_size//2) + x) : int(precise_pred_indexes[i] + x)]

        d_train.append(d_window)
        
        upper_class_index = np.nonzero(upper_half)[0]
        bottom_class_index = np.nonzero(bottom_half)[0]
        
        if (len(upper_class_index) == 0) and (len(bottom_class_index) == 0):
            label_options = [0, 0, 0, 0, 0]
            d_label.append(label_options)
        elif len(upper_class_index) == 0:
            nearest_class = int(bottom_half[bottom_class_index[0]])
            label_options = [0, 0, 0, 0, 0]
            label_options[nearest_class -1] = 1
            d_label.append(label_options)
        elif len(bottom_class_index) == 0:
            nearest_class = int(upper_half[upper_class_index[0]])
            label_options = [0, 0, 0, 0, 0]
            label_options[nearest_class -1] = 1
            d_label.append(label_options)
        elif upper_class_index[0] <= ((win_size//2) - bottom_class_index[0]):
            nearest_class = int(upper_half[upper_class_index[0]])
            label_options = [0, 0, 0, 0, 0]
            label_options[nearest_class -1] = 1
            d_label.append(label_options)
        else:
            nearest_class = int(bottom_half[bottom_class_index[0]])
            label_options = [0, 0, 0, 0, 0]
            label_options[nearest_class -1] = 1
            d_label.append(label_options)

for i in range(int(len(precise_pred_indexes) * 0.8), len(precise_pred_indexes)):
    for x in range(-6, 6):
        
        if (precise_pred_indexes[i] - (win_size//2)) < 0:
            d_window = d1_filtered[0 : int(precise_pred_indexes[i] + (win_size//2) + x)]
            upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : int(precise_pred_indexes[i] + (win_size//2) + x)]
            bottom_half = class_pos_sequence[0: int(precise_pred_indexes[i] + x)]
            avg = np.mean(d_window)
            padding_size = int((win_size//2) - len(bottom_half))
            d_window = np.concatenate([[avg] * padding_size, d_window], axis=0) # * int((win_size//2) - precise_pred_indexes[i])
            bottom_half = np.concatenate([[0] * padding_size, bottom_half], axis=0) 
        elif (precise_pred_indexes[i] + (win_size//2)) > len(d1_filtered):
            d_window = d1_filtered[int(precise_pred_indexes[i] - (win_size//2)) : ]
            upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : ]
            bottom_half = class_pos_sequence[int(precise_pred_indexes[i] - (win_size//2) + x) : int(precise_pred_indexes[i] + x)]
            avg = np.mean(d_window)
            padding_size = int((win_size//2) - len(upper_half))
            d_window = np.concatenate([d_window, [avg] * padding_size ], axis=0)
            upper_half = np.concatenate([upper_half, [0] * padding_size], axis=0)
        else:
            d_window = d1_filtered[int(precise_pred_indexes[i] + x - (win_size//2)) : int(precise_pred_indexes[i] + x + (win_size//2))]
            upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : int(precise_pred_indexes[i] + (win_size//2) + x)]
            bottom_half = class_pos_sequence[int(precise_pred_indexes[i] - (win_size//2) + x) : int(precise_pred_indexes[i] + x)]

        d_val_train.append(d_window)
        
        upper_class_index = np.nonzero(upper_half)[0]
        bottom_class_index = np.nonzero(bottom_half)[0]
        
        if (len(upper_class_index) == 0) and (len(bottom_class_index) == 0):
            label_options = [0, 0, 0, 0, 0]
            d_val_label.append(label_options)
        elif len(upper_class_index) == 0:
            nearest_class = int(bottom_half[bottom_class_index[0]])
            label_options = [0, 0, 0, 0, 0]
            label_options[nearest_class -1] = 1
            d_val_label.append(label_options)
        elif len(bottom_class_index) == 0:
            nearest_class = int(upper_half[upper_class_index[0]])
            label_options = [0, 0, 0, 0, 0]
            label_options[nearest_class -1] = 1
            d_val_label.append(label_options)
        elif upper_class_index[0] <= ((win_size//2) - bottom_class_index[0]):
            nearest_class = int(upper_half[upper_class_index[0]])
            label_options = [0, 0, 0, 0, 0]
            label_options[nearest_class -1] = 1
            d_val_label.append(label_options)
        else:
            nearest_class = int(bottom_half[bottom_class_index[0]])
            label_options = [0, 0, 0, 0, 0]
            label_options[nearest_class -1] = 1
            d_val_label.append(label_options)


d_train = np.array(d_train).reshape(-1, win_size)
d_label = np.array(d_label).reshape(-1, 5)
d_val_train = np.array(d_val_train).reshape(-1, win_size)
d_val_label = np.array(d_val_label).reshape(-1, 5)


model = models.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Normalization(axis=None))
model.add(layers.MaxPooling1D(4))
model.add(layers.Conv1D(20, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(50, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(150, 3, padding="same", activation="sigmoid"))
# model.add(layers.Conv1D(256, 3, padding="same", activation="sigmoid"))
model.add(layers.Flatten())
model.add(layers.Dense(80, activation="sigmoid"))
model.add(layers.Dense(30, activation="sigmoid"))
model.add(layers.Dense(5, activation="sigmoid"))
model.summary()

model.compile(optimizer='adamW', loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])

print("Training Classifier")
history = model.fit(d_train, d_label, epochs=60, batch_size=18, validation_data=(d_val_train, d_val_label)) 

model.save("models/spike_classification_v" + str(model_version) + ".keras")
