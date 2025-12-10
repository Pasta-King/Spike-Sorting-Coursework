import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from detectionEvaluationFunction import indexDelta50Verification


detector_version = 31
second_detect_version = 5
classifier_version = 18

mat = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]

filtered_mat = spio.loadmat("Filtered_Datasets/D1_filtered.mat")
d1_filtered = filtered_mat["re_wave1"]
d1_filtered = d1_filtered[:, 0]

sequence_len = len(d[0])
train_start = 0 


win_size = 50
input_shape = (win_size, 1)
win_step = win_size

d_input = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_window = d1_filtered[i:i + win_size]
    d_input.append(d_window)

d_input = np.array(d_input).reshape(-1, win_size)

print("Detection")
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

    # else:
    #     imprecise_spikes += 1
    #     precise_spike_sequence[(pred_spike_indexes[i] - (win_size//2)) + np.argmax(output[i])] = 1 


precise_pred_indexes = np.nonzero(precise_spike_sequence)[0]


sorted_Index = np.sort(Index[0])

spike_pos_sequence = np.zeros(sequence_len, dtype=np.float64)
for i in range(0, len(Index[0])):
    spike_pos_sequence[Index[0][i]]= 1

total_true_spikes = 0
total_in_range_spikes = 0
total_indexes_matched = 0
total_duplicate_spikes = 0
total_missed_spikes = 0
total_fake_spikes = 0
correctly_classifier = 0
incorrectly_classified = 0
classified_fake_spike = 0


indexDelta50Verification(Index[0], precise_pred_indexes)

# for i in range(0, len(sorted_Index)):
#     if np.any(precise_spike_sequence[sorted_Index[i] - 50: sorted_Index[i] + 50]):
#         total_indexes_matched += 1
#     else:
#         total_missed_spikes += 1

# for i in range(0, len(precise_pred_indexes)):
#     if np.any(spike_pos_sequence[precise_pred_indexes[i] - 50: precise_pred_indexes[i] + 50]):
#         total_in_range_spikes += 1
#     else:
#         total_fake_spikes += 1

# print("total pred spike zones: ", len(pred_spike_indexes))
# print("total pred spikes: ", len(precise_pred_indexes))
# print("total spikes: ", len(sorted_Index))
# print("indexes matched: ", total_indexes_matched)
# print("spikes in range: ", total_in_range_spikes)
# print("missed spikes: ", total_missed_spikes)
# print("fake spikes: ", total_fake_spikes)

print("precise spikes: ", precise_spikes)
print("imprecise spikes: ", imprecise_spikes)