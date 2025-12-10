import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

model_version = 32
classifier_version = 22
dataset_name = "D6.mat"

# mat = spio.loadmat("Coursework-Datasets-20251028/" + dataset_name)
# d = mat["d"]

filtered_mat = spio.loadmat("Filtered_Datasets/D6_filtered.mat")
d1_filtered = filtered_mat["re_wave1"]
d1_filtered = d1_filtered[:, 0]

# Filtering
# def butter_highpass_filter(signal, threshold, freq, order=5):
#     normal_threshold = threshold / (0.5 * freq)
#     b, a = butter(order, normal_threshold, btype="high", analog=False)
#     return filtfilt(b, a, signal)

# d1_filtered = gaussian_filter1d(d[0], 5)
# d1_filtered = butter_highpass_filter(d1_filtered, 5, 25e3)

sequence_len = len(d1_filtered)
train_start = 0


win_size = 200
input_shape = (win_size, 1)
win_step = 160

d_train = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_train.append(d1_filtered[i:i + win_size])

d_train = np.array(d_train).reshape(-1, win_size)

model = keras.models.load_model("models/spike_detection_v" + str(model_version) + ".keras")

output = model.predict(d_train)

x = np.arange(0, sequence_len).tolist()
plt.plot(x, d1_filtered)


relu_flat_output = []

# For Binary
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


# # For SparseCategoricalCrossEntropy
# for i in range (0, len(output)):
#     for x in range(0, win_step):
#         if output[i][x][1] >  output[i][x][0]:
#             prob_window = np.append(0, output[i][x-5:x+5, 1])
#             max_index = np.argmax(prob_window)
#             if max_index == 6:
#                 relu_flat_output.append(1)
#             else:
#                 relu_flat_output.append(0)
#         elif output[i][x][0] >=  output[i][x][1]:
#             relu_flat_output.append(0)
#         else:
#             print("Error spike at: ", len(relu_flat_output))
#             error_spikes += 1
#             relu_flat_output.append(-1)

predicted_spikes = np.nonzero(relu_flat_output)[0]

classifier_model = keras.models.load_model("models/spike_classification_v" + str(classifier_version) + ".keras")

win_size = 100
train_data = []

for i in range(0, len(predicted_spikes)):
    spike_index = predicted_spikes[i]

    if (spike_index + win_size//2 < sequence_len) and (spike_index - win_size//2 >= 0):
        resized_win_sample = d1_filtered[spike_index - win_size//2 : spike_index + win_size//2 ]
    elif ((spike_index + win_size//2 < sequence_len)):
        win_sample = list(d1_filtered[: spike_index + win_size//2 ])
        average = sum(win_sample) / len(win_sample)
        padding = list([average] * int(win_size - len(win_sample)))
        resized_win_sample = padding + win_sample
    elif (spike_index - win_size//2 >= 0):
        win_sample = list(d1_filtered[spike_index - win_size//2 :])
        average = sum(win_sample) / len(win_sample)
        padding = list([average] * int(win_size - len(win_sample)))
        resized_win_sample = win_sample + padding
    else:
        win_sample = list(d1_filtered[:])
        average = sum(win_sample) / len(win_sample)
        padding = list([average] * int(win_size - len(win_sample) // 2))
        resized_win_sample = padding + win_sample + padding
    
    train_data.append(resized_win_sample)

print(len(predicted_spikes))
print(len(train_data))

class_train = np.array(train_data).reshape(-1, win_size) 

classifier_output = classifier_model.predict(class_train)

predicted_classes = []
for i in classifier_output:
    max_prob_index = int(np.argmax(i)) + 1
    predicted_classes.append(max_prob_index)

print(len(predicted_spikes))
print(len(predicted_classes))
# print(predicted_spikes)
# print(predicted_classes)

