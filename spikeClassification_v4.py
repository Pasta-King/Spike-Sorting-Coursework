import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from keras import datasets, layers, models, backend, losses
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
import joblib

model_version = 2
detector_version = 32

def create_feature_matrix (wave, spike_indexes, win_lb=15, win_ub=45):
    feature_matrix = []

    for i in spike_indexes:
        window = wave[max(0, i - win_lb): min(len(wave), i + win_ub)]

        max_amplitude = np.max(window)
        max_amp_index = int(np.argmax(window))
        min_amplitude = np.min(window)
        falling_time = int(np.argmin(window))
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

print("Detection")
detection_model = keras.models.load_model("models/spike_detection_v" + str(detector_version) + ".keras")

output = detection_model.predict(d_input)# [0, :, 0]
    
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

d_train = create_feature_matrix(d[0], precise_pred_indexes)
# d_val_train = create_feature_matrix(d[0], precise_pred_indexes[int(len(precise_pred_indexes) * 0.8):])

win_size = 60
input_shape = (win_size, 1)

class_pos_sequence = np.zeros(sequence_len, dtype=np.float64)
for i in range(0, len(Index[0])):
    class_pos_sequence[Index[0][i]]= Class[0][i]

d_label = []
d_val_label = []

for i in range(0, int(len(precise_pred_indexes))):
        
    if (precise_pred_indexes[i] + x - (win_size//2)) < 0:
        upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : int(precise_pred_indexes[i] + (win_size//2) + x)]
        bottom_half = class_pos_sequence[0: int(precise_pred_indexes[i] + x)]
        padding_size = int((win_size//2) - len(bottom_half))
        bottom_half = np.concatenate([[0] * padding_size, bottom_half], axis=0) 
    elif (precise_pred_indexes[i] + x + (win_size//2)) > len(d1_filtered):
        upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : ]
        bottom_half = class_pos_sequence[int(precise_pred_indexes[i] - (win_size//2) + x) : int(precise_pred_indexes[i] + x)]
        padding_size = int((win_size//2) - len(upper_half))
        upper_half = np.concatenate([upper_half, [0] * padding_size], axis=0)
    else:
        upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : int(precise_pred_indexes[i] + (win_size//2) + x)]
        bottom_half = class_pos_sequence[int(precise_pred_indexes[i] - (win_size//2) + x) : int(precise_pred_indexes[i] + x)]
    
    upper_class_index = np.nonzero(upper_half)[0]
    bottom_class_index = np.nonzero(bottom_half)[0]
    
    if (len(upper_class_index) == 0) and (len(bottom_class_index) == 0):
        label_options = [0, 0, 0, 0, 0]
        # d_label.append(label_options)
        d_label.append(0)
        
    elif len(upper_class_index) == 0:
        nearest_class = int(bottom_half[bottom_class_index[0]])
        label_options = [0, 0, 0, 0, 0]
        label_options[nearest_class -1] = 1
        d_label.append(nearest_class) # d_label.append(label_options)
    elif len(bottom_class_index) == 0:
        nearest_class = int(upper_half[upper_class_index[0]])
        label_options = [0, 0, 0, 0, 0]
        label_options[nearest_class -1] = 1
        d_label.append(nearest_class)
    elif upper_class_index[0] <= ((win_size//2) - bottom_class_index[0]):
        nearest_class = int(upper_half[upper_class_index[0]])
        label_options = [0, 0, 0, 0, 0]
        label_options[nearest_class -1] = 1
        d_label.append(nearest_class)
    else:
        nearest_class = int(bottom_half[bottom_class_index[0]])
        label_options = [0, 0, 0, 0, 0]
        label_options[nearest_class -1] = 1
        d_label.append(nearest_class)

# for i in range(int(len(precise_pred_indexes) * 0.8), len(precise_pred_indexes)):
        
#     if (precise_pred_indexes[i] - (win_size//2)) < 0:
#         upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : int(precise_pred_indexes[i] + (win_size//2) + x)]
#         bottom_half = class_pos_sequence[0: int(precise_pred_indexes[i] + x)]
#         padding_size = int((win_size//2) - len(bottom_half))
#         bottom_half = np.concatenate([[0] * padding_size, bottom_half], axis=0) 
#     elif (precise_pred_indexes[i] + (win_size//2)) > len(d1_filtered):
#         upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : ]
#         bottom_half = class_pos_sequence[int(precise_pred_indexes[i] - (win_size//2) + x) : int(precise_pred_indexes[i] + x)]
#         padding_size = int((win_size//2) - len(upper_half))
#         upper_half = np.concatenate([upper_half, [0] * padding_size], axis=0)
#     else:
#         upper_half = class_pos_sequence[int(precise_pred_indexes[i] + x) : int(precise_pred_indexes[i] + (win_size//2) + x)]
#         bottom_half = class_pos_sequence[int(precise_pred_indexes[i] - (win_size//2) + x) : int(precise_pred_indexes[i] + x)]
    
#     upper_class_index = np.nonzero(upper_half)[0]
#     bottom_class_index = np.nonzero(bottom_half)[0]
    
#     if (len(upper_class_index) == 0) and (len(bottom_class_index) == 0):
#         label_options = [0, 0, 0, 0, 0]
#         d_val_label.append(0)
#     elif len(upper_class_index) == 0:
#         nearest_class = int(bottom_half[bottom_class_index[0]])
#         label_options = [0, 0, 0, 0, 0]
#         label_options[nearest_class -1] = 1
#         d_val_label.append(nearest_class)
#     elif len(bottom_class_index) == 0:
#         nearest_class = int(upper_half[upper_class_index[0]])
#         label_options = [0, 0, 0, 0, 0]
#         label_options[nearest_class -1] = 1
#         d_val_label.append(nearest_class)
#     elif upper_class_index[0] <= ((win_size//2) - bottom_class_index[0]):
#         nearest_class = int(upper_half[upper_class_index[0]])
#         label_options = [0, 0, 0, 0, 0]
#         label_options[nearest_class -1] = 1
#         d_val_label.append(nearest_class)
#     else:
#         nearest_class = int(bottom_half[bottom_class_index[0]])
#         label_options = [0, 0, 0, 0, 0]
#         label_options[nearest_class -1] = 1
#         d_val_label.append(nearest_class)



clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, class_weight="balanced")

print("Random Tree Classifier")
clf.fit(d_train, d_label)

pred_class = clf.predict(d_train)

counts = np.bincount(pred_class, minlength=6)[1:]
print(counts)

joblib.dump(clf, "models/random_tree_classifier_v" + str(model_version) + ".keras")

# d_train = np.array(d_train).reshape(-1, win_size)
# d_label = np.array(d_label).reshape(-1, 5)
# d_val_train = np.array(d_val_train).reshape(-1, win_size)
# d_val_label = np.array(d_val_label).reshape(-1, 5)


# model = models.Sequential()
# model.add(layers.Input(shape=input_shape))
# model.add(layers.Normalization(axis=None))
# model.add(layers.MaxPooling1D(4))
# model.add(layers.Conv1D(20, 3, padding="same", activation="sigmoid"))
# model.add(layers.Conv1D(50, 3, padding="same", activation="sigmoid"))
# model.add(layers.Conv1D(150, 3, padding="same", activation="sigmoid"))
# # model.add(layers.Conv1D(256, 3, padding="same", activation="sigmoid"))
# model.add(layers.Flatten())
# model.add(layers.Dense(80, activation="sigmoid"))
# model.add(layers.Dense(30, activation="sigmoid"))
# model.add(layers.Dense(5, activation="sigmoid"))
# model.summary()

# model.compile(optimizer='adamW', loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])

# print("Training Classifier")
# history = model.fit(d_train, d_label, epochs=60, batch_size=18, validation_data=(d_val_train, d_val_label)) 

# model.save("models/random_tree_classifier_v" + str(model_version) + ".keras")
