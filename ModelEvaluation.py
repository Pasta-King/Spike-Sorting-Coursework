"""ModelEvaluation.py contains the sequence used to evaluate the detection and classification models

Loads the filtered D1 dataset, formats the data and passes it to the detection model. to predict the location of the spikes. 
The output of the detection model is then formatted and passed to the classification model to predict the class of each 
spike. The results of each model is then compared with the Index and Classes to evaluate the performance of each model.

Dependencies:
- numpy
- keras
"""

import numpy as np
import keras
from DataProcessing import load_dataset, format_detection_input, format_classifier_input, classifier_post_processing



detection_version = 1
classifier_version = 1

detect_win_size = 50
detect_win_step = 30
classify_win_size = 100


# Getting Predictions
filtered_data = load_dataset("D1_Noisy_filtered.mat", labelled=False, original=False)

detection_input_data, _ = format_detection_input(filtered_data, 0, len(filtered_data), detect_win_size, detect_win_step)

detector_model = keras.models.load_model("models/spike_detection_v" + str(detection_version) + ".keras")

print("Detecting Spike Indexes")
output = detector_model.predict(detection_input_data)

# Similar code to detection_post_processing is used, as we need to extract the relu_flat_output for evaluation
relu_flat_output = []
# Stepping through each window and each index in that window
for i in range (0, len(output)):
    for x in range(0, detect_win_step):
        if output[i][x] >  0.5:
            # Finding the index with the highest probability of being a spike within a window of 20
            prob_window = np.append(0, output[i][x-10:x+10]) # Window is added to 0 to prevent an empty array
            max_index = np.argmax(prob_window)
            if max_index == 11:
                # If the current index is the index with the highest probability of being a spike in the window, the spike is recorded
                relu_flat_output.append(1)
            else:
                relu_flat_output.append(0)
        else:
            relu_flat_output.append(0)

# Finds the index of every spike recorded returns it
predicted_spikes = np.nonzero(relu_flat_output)[0]


classifier_input_data = format_classifier_input(predicted_spikes, filtered_data, classify_win_size)

classifier_model = keras.models.load_model("models/spike_classification_v" + str(classifier_version) + ".keras")

classifier_output = classifier_model.predict(classifier_input_data)

predicted_classes = classifier_post_processing(classifier_output)


# Evaluation Predictions
d1_noisy, d1_indexes, d1_classes = load_dataset("D1.mat", labelled=True, original=True)

# Sequence created show which indexes in the neuron recording have a spike 
label_spike_sequence = np.zeros(len(filtered_data), dtype=np.float64)
for i in range(0, len(d1_indexes)):
    label_spike_sequence[d1_indexes[i]] = 1

total_indexes_matched = 0
total_missed_indexes = 0
total_in_range_spikes = 0
total_fake_spikes = 0

# Checks that there is at least one predicted spike within the range of each real spike
for i in range(0, len(d1_indexes)):
    if np.any(relu_flat_output[d1_indexes[i] - 50: d1_indexes[i] + 50]):
        total_indexes_matched += 1
    else:
        total_missed_indexes += 1

# Checks that every predicted spike is within range of a real spike
for i in range(0, len(predicted_spikes)):
    if np.any(label_spike_sequence[predicted_spikes[i] - 50: predicted_spikes[i] + 50]):
        total_in_range_spikes += 1
    else:
        total_fake_spikes += 1

print("total pred spikes: ", len(predicted_spikes))
print("total spikes: ", len(d1_indexes))
print("indexes matched: ", total_indexes_matched)
print("missed indexes: ", total_missed_indexes)
print("spikes in range: ", total_in_range_spikes)
print("fake spikes: ", total_fake_spikes)

# Shows the number of each class predicted vs the actual number of each class in the dataset
print("Predicted Class count: ")
print(np.unique(predicted_classes, return_counts=True))
print("Actual Class count: ")
print(np.unique(d1_classes, return_counts=True))