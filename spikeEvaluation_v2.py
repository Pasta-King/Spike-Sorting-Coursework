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
classifier_version = 21

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

class_pos_sequence = np.zeros(sequence_len, dtype=np.float64)
for i in range(0, len(Index[0])):
    class_pos_sequence[Index[0][i]]= Class[0][i]


print("Classifier")

classifier_model = keras.models.load_model("models/spike_classification_v" + str(classifier_version) + ".keras")
classifier_model.summary()

win_size = 200
class_train = []

for i in range(0, int(len(precise_pred_indexes))):
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

        class_train.append(d_window)

class_train = np.array(class_train).reshape(-1, win_size)

classifier_output = classifier_model.predict(class_train)

predicted_classes = []
for i in classifier_output:
    max_prob_index = np.argmax(i) 
    predicted_classes.append(max_prob_index)

for i in range(0, len(Index[0])):
    spike_index = Index[0][i]
    
    plt.plot(spike_index, d1_filtered[spike_index], "kx")

for i in range(0, len(precise_pred_indexes)):
    true_spike = 0
    in_range_spike = 0
    true_class = 8000

    for x in range(0, len(Index[0])):
        if precise_pred_indexes[i] == Index[0][x]:
            plt.vlines(precise_pred_indexes[i], 0, 4, colors="g")
            total_true_spikes += 1
            true_spike = 1
            true_class = Class[0][x]
    
    if true_spike == 0:
        for x in range(0, len(Index[0])):
            if (precise_pred_indexes[i] < Index[0][x] + 50) and (precise_pred_indexes[i] > Index[0][x] - 50):
                if in_range_spike == 0:
                    total_in_range_spikes += 1
                    in_range_spike = 1
                    plt.vlines(precise_pred_indexes[i], 0, 4, colors="m")
                    true_class = Class[0][x]
                else:
                    total_duplicate_spikes += 1
                    

    if true_class != 8000:
        if true_class == predicted_classes[i]:
            correctly_classifier += 1
        else:
            incorrectly_classified += 1
    else:
        classified_fake_spike += 1

    
    if (true_spike == 0) and (in_range_spike == 0):
        plt.vlines(i, 0, 4, colors="r")
        total_fake_spikes += 1

plt.show()
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
print("total classified: ", len(predicted_classes))
print("correctly classified: ", correctly_classifier)
print("incorrectly classified: ", incorrectly_classified)
print("fake classified: ", classified_fake_spike)
