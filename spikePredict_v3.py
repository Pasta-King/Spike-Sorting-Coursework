import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
import joblib


detector_version = 32
tree_classifier_version = 2
dataset_name = "D2.mat"

filtered_mat = spio.loadmat("Filtered_Datasets/D2_filtered.mat")
d1_filtered = filtered_mat["re_wave1"]
d1_filtered = d1_filtered[:, 0]

def create_feature_matrix (wave, spike_indexes, win_lb=15, win_ub=45):
    feature_matrix = []

    for i in spike_indexes:
        window = wave[max(0, i - win_lb): min(len(wave), i + win_ub)]

        max_amplitude = np.max(window)
        max_amp_index = int(np.argmax(window))
        min_amplitude = np.min(window)
        wave_width = np.trapezoid(window, dx=1)
        amplitude_diff = max_amplitude - min_amplitude
        spike_location_normalized = max_amp_index/ max(1, win_lb + win_ub - 1)
        root_mean_square = np.sqrt(np.mean(window ** 2))
        zeroth_intersections = float(np.mean(np.abs(np.diff(np.sign(window))) > 0))
        skew_value = float(skew(window))
        kurt_value = float(kurtosis(window))
        
        rising_gradient = (window[max_amp_index] - window[0]) / max(1, max_amp_index)
        falling_gradient = (window[-1] - window[max_amp_index]) / max(1, (win_ub + win_lb -1 - max_amp_index))
        
        feature_matrix.append([max_amplitude, min_amplitude, wave_width, amplitude_diff, spike_location_normalized, root_mean_square, zeroth_intersections, skew_value, kurt_value, rising_gradient, falling_gradient])

    feature_matrix = np.array(feature_matrix)
    return feature_matrix

# mat = spio.loadmat("Coursework-Datasets-20251028/" + dataset_name)
# d = mat["d"]



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

predicted_spikes = np.nonzero(relu_flat_output)[0]

d_train = create_feature_matrix(d1_filtered, predicted_spikes)

tree_classifier = joblib.load("models/random_tree_classifier_v" + str(tree_classifier_version) + ".keras")

classifier_output = tree_classifier.predict(d_train)

counts = np.bincount(classifier_output, minlength=6)[1:]
print(counts)


print("pred indexes: ", len(predicted_spikes))
print("pred class output", len(classifier_output))

# print(predicted_spikes)
# print(predicted_classes)

mat_dict = {"Index": predicted_spikes, "Class": classifier_output}
spio.savemat("results/" + dataset_name, mat_dict)
