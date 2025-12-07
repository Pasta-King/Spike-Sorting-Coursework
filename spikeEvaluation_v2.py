import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d


detector_version = 26
second_detect_version = 1
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

detector_model.summary()

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


precise_spike_sequence = np.zeros(sequence_len, dtype=np.float64)

second_detect_model = keras.models.load_model("models/spike_detect_second_stage_v" + str(second_detect_version) + ".keras")

for i in range(0, len(pred_spike_indexes)):
    d_window = d1_filtered[int(pred_spike_indexes[i]) - 37 : int(pred_spike_indexes[i]) + 38]
    d_input = np.array(d_window).reshape(-1, 75)
    
    output = second_detect_model.predict([d_input])

    if np.any(output[0]):
        for x in np.nonzero(output[0])[0]:
            precise_spike_sequence[(pred_spike_indexes[i] - 12) + x] = 1


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


for i in range(0, len(sorted_Index)):
    if np.any(spike_pos_sequence[sorted_Index[i] - 50: sorted_Index[i] + 50]):
        total_indexes_matched += 1
    else:
        total_missed_spikes += 1

for i in range(0, len(pred_spike_indexes)):
    if np.any(spike_pos_sequence[pred_spike_indexes[i] - 50: pred_spike_indexes[i] + 50]):
        total_in_range_spikes += 1
    else:
        total_fake_spikes += 1

print("total pred spikes: ", len(pred_spike_indexes))
print("total spikes: ", len(sorted_Index))
print("indexes matched: ", total_indexes_matched)
print("spikes in range: ", total_in_range_spikes)
print("missed spikes: ", total_missed_spikes)
print("fake spikes: ", total_fake_spikes)

# # Going through predicted spikes and checking accuracy
# predicted_spikes = np.nonzero(relu_flat_output)[0]
# adjusted_predicted_spikes = []

# classifier_model = keras.models.load_model("models/spike_classification_v" + str(classifier_version) + ".keras")

# win_size = 100
# train_data = []

# for i in range(0, len(predicted_spikes)):
#     spike_index = predicted_spikes[i]

#     local_peaks, properties = find_peaks(d1_filtered[spike_index-10:spike_index+10], height=0.2)
#     if len(local_peaks) <= 1:
#         adjusted_predicted_spikes.append(spike_index)
#     else:
#         for x in range(0, len(local_peaks)):
#             adjusted_predicted_spikes.append(spike_index + x * 3)


# for i in range(0, len(adjusted_predicted_spikes)):
#     spike_index = adjusted_predicted_spikes[i]

#     if (spike_index + win_size//2 < sequence_len) and (spike_index - win_size//2 >= 0):
#         train_data.append(d1_filtered[spike_index - win_size//2 : spike_index + win_size//2 ])
#     elif ((spike_index + win_size//2 < sequence_len)):
#         train_data.append(d1_filtered[: spike_index + win_size//2 ])
#     elif (spike_index - win_size//2 >= 0):
#         train_data.append(d1_filtered[spike_index - win_size//2 :])
#     else:
#         train_data.append(d1_filtered[:])

# class_train = np.array(train_data).reshape(-1, win_size) 

# classifier_output = classifier_model.predict(class_train)

# predicted_classes = []
# for i in classifier_output:
#     max_prob_index = np.argmax(i) + 1
#     predicted_classes.append(max_prob_index)

# for i in range(0, len(Index[0])):
#     spike_index = Index[0][i]
    
#     plt.plot(spike_index, d1_filtered[spike_index], "kx")

# for i in range(0, len(predicted_spikes)):
#     true_spike = 0
#     in_range_spike = 0
#     true_class = 8000

#     for x in range(0, len(Index[0])):
#         if predicted_spikes[i] == Index[0][x]:
#             plt.vlines(predicted_spikes[i], 0, 4, colors="g")
#             total_true_spikes += 1
#             true_spike = 1
#             true_class = Class[0][x]
    
#     if true_spike == 0:
#         for x in range(0, len(Index[0])):
#             if (predicted_spikes[i] < Index[0][x] + 50) and (predicted_spikes[i] > Index[0][x] - 50):
#                 if in_range_spike == 0:
#                     total_in_range_spikes += 1
#                     in_range_spike = 1
#                     plt.vlines(predicted_spikes[i], 0, 4, colors="m")
#                     true_class = Class[0][x]
#                 else:
#                     total_duplicate_spikes += 1
                    

#     if true_class != 8000:
#         if true_class == predicted_classes[i]:
#             correctly_classifier += 1
#         else:
#             incorrectly_classified += 1
#     else:
#         classified_fake_spike += 1

    
#     if (true_spike == 0) and (in_range_spike == 0):
#         plt.vlines(i, 0, 4, colors="r")
#         total_fake_spikes += 1

# missing_spikes = 0

# for i in range(0, len(Index[0])):
#     for x in predicted_spikes:
#         spike_present = 0
#         if (abs(Index[0][i] - x) < 50):
#             spike_present = 1

#     if spike_present == 0:
#         missing_spikes += 1
    

# print("total spikes: ", len(Index[0]))
# print("total predicted spikes:", len(predicted_spikes))
# print("true spikes: ", total_true_spikes)
# print("spikes in range: ", total_in_range_spikes)
# print("in multiple ranges: ", total_duplicate_spikes)
# print("missing spikes: ", len(Index[0]) - (total_true_spikes + total_in_range_spikes)) # len(Index[0]) - (total_true_spikes + total_in_range_spikes)
# print("error spikes: ", error_spikes)
# print("fake spikes: ", total_fake_spikes)
# print("total classified: ", len(predicted_classes))
# print("correctly classified: ", correctly_classifier)
# print("incorrectly classified: ", incorrectly_classified)
# print("fake classified: ", classified_fake_spike)

# plt.show()
