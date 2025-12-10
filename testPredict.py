import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d


detector_version = 31
second_detect_version = 5
classifier_version = 19
dataset_name = "D2.mat"

# mat = spio.loadmat("Coursework-Datasets-20251028/" + dataset_name)
# d = mat["d"]

filtered_mat = spio.loadmat("Filtered_Datasets/D2_filtered.mat")
d1_filtered = filtered_mat["re_wave1"]
d1_filtered = d1_filtered[:, 0]

sequence_len = len(d1_filtered)
train_start = 0 

win_size = 50
input_shape = (win_size, 1)
win_step = win_size

d_input = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_window = d1_filtered[i:i + win_size]
    d_input.append(d_window)

d_input = np.array(d_input).reshape(-1, win_size)


detector_model = keras.models.load_model("models/spike_detection_v" + str(detector_version) + ".keras")

# detector_model.summary()
print("Detection")
output = detector_model.predict(d_input)

x = np.arange(0, sequence_len).tolist()
plt.plot(x, d1_filtered)


relu_flat_output = []
pred_spike_indexes = []
error_spikes = 0

# For Binary
for i in range (0, len(output)):
    if output[i] >= 0.5:
        if win_size // 2:
            output_window = [0] * (win_size//2) + [1] + [0] * ((win_size-1)//2)
        else: 
            output_window = [0] * (win_size//2) + [1] + [0] * (win_size//2)
        relu_flat_output = relu_flat_output + output_window

        pred_spike_indexes.append((i * win_size) + (win_size//2))
    
    else:
        output_window = [0] * win_size
        relu_flat_output = relu_flat_output + output_window


precise_spike_sequence = np.zeros(sequence_len, dtype=np.float64)

second_detect_model = keras.models.load_model("models/spike_detect_second_stage_v" + str(second_detect_version) + ".keras")

d_input = []
for i in range(0, len(pred_spike_indexes)):
    d_window = d1_filtered[int(pred_spike_indexes[i]) - (win_size//2) : int(pred_spike_indexes[i]) + (win_size//2)]
    d_input.append(d_window)

d_input = np.array(d_input).reshape(-1, win_size)


second_detect_model = keras.models.load_model("models/spike_detect_second_stage_v" + str(second_detect_version) + ".keras")

print("Second Stage")
output = second_detect_model.predict(d_input)# [0, :, 0]
    
precise_spikes = 0
imprecise_spikes = 0

for i in range(0, len(output)):
    output_window = output[i]

    win_pred_spike = np.nonzero(output_window > 0.5)[0]

    if len(win_pred_spike) > 0:
        for x in win_pred_spike:
            if np.all(output_window[x-1: x+2] > 0.5):
                precise_spikes += 1
                precise_spike_sequence[(pred_spike_indexes[i] - (win_size//2)) + x] = 1

    else:
        imprecise_spikes += 1
        precise_spike_sequence[(pred_spike_indexes[i] - (win_size//2)) + np.argmax(output[i])] = 1 


precise_pred_indexes = np.nonzero(precise_spike_sequence)[0]

pred_len = len(precise_pred_indexes)
train_start = 0

win_size = 100
input_shape = (win_size, 1)

d_data = []

for i in range(0, int(len(precise_pred_indexes))):
    if (precise_pred_indexes[i] - (win_size//2)) < 0:
        d_window = d1_filtered[0 : int(precise_pred_indexes[i] + (win_size//2))]
        avg = np.mean(d_window)
        d_window = np.concatenate([[avg] * int((win_size//2) - precise_pred_indexes[i]), d_window], axis=0) # * int((win_size//2) - precise_pred_indexes[i])
    elif (precise_pred_indexes[i] + (win_size//2)) > sequence_len:
        d_window = d1_filtered[int(precise_pred_indexes[i] - (win_size//2)) : ]
        avg = np.mean(d_window)

        d_window = np.concatenate([d_window, [avg] * int((win_size//2) + precise_pred_indexes[i] - sequence_len)], axis=0)
    else:
        d_window = d1_filtered[int(precise_pred_indexes[i] - (win_size//2)) : int(precise_pred_indexes[i] + (win_size//2))]
    
    d_data.append(d_window)

d_data = np.array(d_data).reshape(-1, win_size)
print("Classifier")
classifier_model = keras.models.load_model("models/spike_classification_v" + str(classifier_version) + ".keras")

classifier_output = classifier_model.predict(d_data)

print("pred indexes: ", len(precise_pred_indexes))
print("pred class output", len(classifier_output))

predicted_classes = []
adjusted_predicted_spikes = []
for i in range(0, len(classifier_output)):
    max_prob_index = int(np.argmax(classifier_output[i]))

    if max_prob_index > 0:
        predicted_classes.append(max_prob_index)
        adjusted_predicted_spikes.append(precise_pred_indexes[i])

print(len(predicted_classes))
print("precise spikes: ", precise_spikes)
print("imprecise spikes: ", imprecise_spikes)
# print(predicted_spikes)
# print(predicted_classes)