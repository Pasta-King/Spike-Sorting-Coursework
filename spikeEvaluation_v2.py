import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d


detector_version = 20
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


win_size = 200
input_shape = (win_size, 1)
win_step = 160

d_input = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_window = d1_filtered[i:i + win_size]
    #noise = np.random.normal(0, 1, [win_size]) 
    d_input.append(d_window)

d_input = np.array(d_input).reshape(-1, win_size)

detector_model = keras.models.load_model("models/spike_detection_v" + str(detector_version) + ".keras")

detector_model.summary()

output = detector_model.predict(d_input)

x = np.arange(0, sequence_len).tolist()
plt.plot(x, d1_filtered)


relu_flat_output = []
error_spikes = 0

# For BinaryCrossEntropy
#for i in range (0, len(output)):
#    relu_layer = [1 if x > 0.5 else 0 for x in output[i][:160]]
#    relu_flat_output = relu_flat_output + relu_layer

# For SparseCategoricalCrossEntropy
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

# for i in range (0, len(output)):
#     for x in range(0, win_step):
#         if output[i][x] >  0.4:
#             if np.all(output[i][x: x + 4] > [0.4, 0.4, 0.4, 0.4]):
#                     relu_flat_output.append(1)
#                     x += 4
#         else:
#             relu_flat_output.append(0)


# for i in range (0, len(output)):
#     for x in range(0, win_step):
#         if output[i][x] >  0.5:
#             relu_flat_output.append(1)
#         else:
#             relu_flat_output.append(0)

#print(len(output))


total_true_spikes = 0
total_in_range_spikes = 0
total_duplicate_spikes = 0
total_fake_spikes = 0
correctly_classifier = 0
incorrectly_classified = 0
classified_fake_spike = 0

# Going through predicted spikes and checking accuracy
predicted_spikes = np.nonzero(relu_flat_output)[0]
adjusted_predicted_spikes = []

classifier_model = keras.models.load_model("models/spike_classification_v" + str(classifier_version) + ".keras")

win_size = 100
train_data = []

for i in range(0, len(predicted_spikes)):
    spike_index = predicted_spikes[i]

    local_peaks, properties = find_peaks(d1_filtered[spike_index-10:spike_index+10], height=0.2)
    if len(local_peaks) <= 1:
        adjusted_predicted_spikes.append(spike_index)
    else:
        for x in range(0, len(local_peaks)):
            adjusted_predicted_spikes.append(spike_index + x * 3)


for i in range(0, len(adjusted_predicted_spikes)):
    spike_index = adjusted_predicted_spikes[i]

    if (spike_index + win_size//2 < sequence_len) and (spike_index - win_size//2 >= 0):
        train_data.append(d1_filtered[spike_index - win_size//2 : spike_index + win_size//2 ])
    elif ((spike_index + win_size//2 < sequence_len)):
        train_data.append(d1_filtered[: spike_index + win_size//2 ])
    elif (spike_index - win_size//2 >= 0):
        train_data.append(d1_filtered[spike_index - win_size//2 :])
    else:
        train_data.append(d1_filtered[:])

class_train = np.array(train_data).reshape(-1, win_size) 

classifier_output = classifier_model.predict(class_train)

predicted_classes = []
for i in classifier_output:
    max_prob_index = np.argmax(i) + 1
    predicted_classes.append(max_prob_index)

for i in range(0, len(Index[0])):
    spike_index = Index[0][i]
    
    plt.plot(spike_index, d1_filtered[spike_index], "kx")

for i in range(0, len(predicted_spikes)):
    true_spike = 0
    in_range_spike = 0
    true_class = 8000

    for x in range(0, len(Index[0])):
        if predicted_spikes[i] == Index[0][x]:
            plt.vlines(predicted_spikes[i], 0, 4, colors="g")
            total_true_spikes += 1
            true_spike = 1
            true_class = Class[0][x]
    
    if true_spike == 0:
        for x in range(0, len(Index[0])):
            if (predicted_spikes[i] < Index[0][x] + 50) and (predicted_spikes[i] > Index[0][x] - 50):
                if in_range_spike == 0:
                    total_in_range_spikes += 1
                    in_range_spike = 1
                    plt.vlines(predicted_spikes[i], 0, 4, colors="m")
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

missing_spikes = 0

for i in range(0, len(Index[0])):
    for x in predicted_spikes:
        spike_present = 0
        if (abs(Index[0][i] - x) < 50):
            spike_present = 1

    if spike_present == 0:
        missing_spikes += 1
    

print("total spikes: ", len(Index[0]))
print("total predicted spikes:", len(predicted_spikes))
print("true spikes: ", total_true_spikes)
print("spikes in range: ", total_in_range_spikes)
print("in multiple ranges: ", total_duplicate_spikes)
print("missing spikes: ", len(Index[0]) - (total_true_spikes + total_in_range_spikes)) # len(Index[0]) - (total_true_spikes + total_in_range_spikes)
print("error spikes: ", error_spikes)
print("fake spikes: ", total_fake_spikes)
print("total classified: ", len(predicted_classes))
print("correctly classified: ", correctly_classifier)
print("incorrectly classified: ", incorrectly_classified)
print("fake classified: ", classified_fake_spike)

plt.show()