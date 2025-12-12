"""PredictingResults.py contains the sequence used to produce a .mat file with the predicted Index and Class values for all the dataset

Loops through the datasets. Loads the filtered neuron recording of that dataset. Formats and passes that data to the detection model 
to predict the location of the spikes. The output of the detection model is then formatted and passed to the classification model to 
predict the class of each spike. The results of both are then saved to results/ as a .mat file with the name of the dataset

Dependencies:
- scipy
- keras
"""

import scipy.io as spio
import keras
from DataProcessing import load_dataset, format_detection_input, detection_post_processing, format_classifier_input, classifier_post_processing



detection_version = 1
classifier_version = 1

detect_win_size = 50
detect_win_step = 30
classify_win_size = 100

# The name of the datasets to iterate over
dataset_names = ["D2", "D3", "D4", "D5", "D6"]


for filename in dataset_names:
    filtered_data = load_dataset(filename + "_filtered.mat", labelled=False, original=False)

    detection_input_data, _ = format_detection_input(filtered_data, 0, len(filtered_data), detect_win_size, detect_win_step)

    detector_model = keras.models.load_model("models/spike_detection_v" + str(detection_version) + ".keras")

    print("Detecting spike indexes")
    output = detector_model.predict(detection_input_data)

    predicted_spikes = detection_post_processing(output, detect_win_step)

    classifier_input_data = format_classifier_input(predicted_spikes, filtered_data, classify_win_size)

    classifier_model = keras.models.load_model("models/spike_classification_v" + str(classifier_version) + ".keras")

    print("Classifying the spikes")
    classifier_output = classifier_model.predict(classifier_input_data)

    predicted_classes = classifier_post_processing(classifier_output)

    mat_dict = {"Index": predicted_spikes, "Class": predicted_classes}

    # Predictions are saved in the results/ folder
    spio.savemat("results/" + filename + ".mat", mat_dict)
