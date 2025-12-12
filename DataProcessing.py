"""DataProcessing.py contains the functions that are frequently used to process and format data in other files.

This is the file that contains the functions imported by the other files. It only contains the function code 
and no mainline code, and will not produce an effect if the file is executed. 

Dependencies:
- numpy
- scipy
"""

import numpy as np
import scipy.io as spio



def load_dataset(filename, labelled=False, original=True):
    """Returns the data stored in the file given by filename

    Params:
    - name: String, the filname with the data to be accessed
    - labelled: Boolean, default is False
                True to retrieve the Index and Class data alongside the neuron recording
                False to retrieve just the neuron recording
    - original: Boolean, default is True
                True if the original, unfiltered data is being accessed
                False if the filtered datasets are being accessed
    """

    if original == True:
        mat = spio.loadmat("Coursework-Datasets/" + filename)
        d = mat["d"][0]  # Converts the retrieved matrix into a 1D vector
    else:
        mat = spio.loadmat("Filtered_Datasets/" + filename)
        d = mat["re_wave1"][:, 0]  # Converts the retrieved matrix into a 1D vector

    if labelled == False:
        return d
    else:
        Index = mat["Index"]
        Class = mat["Class"]
        return d, Index[0], Class[0]


def save_wave(wave, filename, indexes=[], classes=[]):
    """Saves the data to a file given by the filename

    Params:
    - wave: Array, the neuron spike data that will be saved as "d" in the file
    - filename: String, the file name the data will be saved to
    - indexes: Array, default is an empty array
                If the Array is non-empty then both the indexes and classes data is saved
                in the file as Index and Class respectively
    - classes: Array, default is an empty array
                data is stored in the file as Class if indexes parameter is a non-empty array
    """

    if len(indexes) > 0:
        m_dict = {"d": wave, "Index": indexes, "Class": classes}
    else:
        m_dict = {"d": wave}

    spio.savemat(filename, mdict=m_dict)


def create_spike_sequence(spike_indexes, sequence_len):
    """Returns an array with tappering probability around the spike indexes

    Params:
    - spike_indexes: Array, containing the indexes where spikes are present in the neuron recording
    - sequence_len: Integer, the length of the neuron recording
    """
    spike_sequence = np.zeros(sequence_len, dtype=np.float64)

    for i in range(0, len(spike_indexes)):
        spike_sequence[spike_indexes[i] - 4: spike_indexes[i] + 5] = [0.7, 0.75, 0.8, 0.85, 1, 0.85, 0.8, 0.75, 0.7]

    return spike_sequence


def create_class_sequence(spike_indexes, class_labels, sequence_len):
    """Returns an array with the class values at the indexes of the spikes in the neuron recording

    Params:
    - spike_indexes: Array, containing the indexes where spikes are present in the neuron recording
    - class_labels: Array, containing the class of the spike in the corresponding index in spike_indexes
    - sequence_len: Integer, the length of the neuron recording
    """

    class_sequence = np.zeros(sequence_len, dtype=np.float64)
    
    # Each spike_index[i] corresponds to a class_label[i]
    for i in range(0, len(spike_indexes)):
        class_sequence[spike_indexes[i]] = class_labels[i]
    
    return class_sequence


def format_detection_input(
        data_sequence, start, end, window_size, offset, 
        label_sequence=[]):
    """Returns the training_data and label_data formatted into the correct shape

    Params:
    - data_sequence: Array, containing the neuron recording
    - start: Integer, the index of the data_sequence it should begin formatting
    - end: Integer, the index of the the data_sequence it should end formatting
    - window_size: Integer, the size of the window that will be input into the model
    - offset: Integer, the step between the start of the first window to the start of the next window
    - label_sequence: Array, default empty
                        If the Array is empty then only a window of the data_sequence will be created and appended 
                        to training_data, label_data will remain empty
                        If the Array is the same length as data_sequence then both a window of the data sequence 
                        and the Array will be created and appended to training_data and label_data respectively
                        If the Array is non-empty and not the same length as data_sequence then something has 
                        gone wrong so an IndexError is raised
    """

    training_data = []
    label_data = []

    if len(label_sequence) == 0:
        for i in range(start, end - offset, offset):
            # label data is left empty
            training_data.append(data_sequence[i:i + window_size])

    elif len(label_sequence) == len(data_sequence):
        for i in range(start, end - offset, offset):
            training_data.append(data_sequence[i:i + window_size])
            label_data.append(label_sequence[i:i + window_size])
    else:
        # If the program reaches here something is wrong
        raise IndexError
        
    training_data = np.array(training_data).reshape(-1, window_size)
    label_data = np.array(label_data)

    return training_data, label_data


def get_detection_training_data(win_size, win_step):
    """Returns the train_data, label_data, val_train_data, val_label_data that will be passed to the detection model to train

    Params:
    - win_size: Integer, the length of the window that the training data is formatted to fit into
    - win_step: Integer, the step between the start of the first window to the start of the next window
    """

    d1_noisy, d1_indexes, d1_classes = load_dataset("D1.mat", labelled=True, original=True)
    d1_filtered = load_dataset("D1_Noisy_filtered.mat", labelled=False, original=False)

    d1_labels = create_spike_sequence(d1_indexes, len(d1_filtered))

    train_start = 0
    train_end = int(len(d1_filtered) * 0.8)

    train_data, label_data = format_detection_input(d1_filtered, train_start, train_end, win_size, win_step, d1_labels)
    val_train_data, val_label_data = format_detection_input(d1_filtered, train_end, len(d1_filtered), win_size, win_step, d1_labels)

    return train_data, label_data, val_train_data, val_label_data


def get_full_detection_data(win_size, win_step):
    """Returns train_data, label_data that contains the full length of the D1 neuron recording, no validation data

    Params:
    - win_size: Integer, the length of the window that the training data is formatted to fit into
    - win_step: Integer, the step between the start of the first window to the start of the next window
    """

    d1_noisy, d1_indexes, d1_classes = load_dataset("D1.mat", labelled=True, original=True)
    d1_filtered = load_dataset("D1_Noisy_filtered.mat", labelled=False, original=False)

    d1_labels = create_spike_sequence(d1_indexes, len(d1_filtered))

    train_start = 0
    train_end = len(d1_filtered)

    train_data, label_data = format_detection_input(d1_filtered, train_start, train_end, win_size, win_step, d1_labels)

    return train_data, label_data


def detection_post_processing(output, win_step):
    """Returns a list with the index of every spike predicted by the model

    Params:
    - output: 2D Array, the output of the detection model, split into windows
    - win_step: Integer, the difference between the start of one window and the start of the next
    """

    relu_flat_output = []
    # Stepping through each window and each index in that window
    for i in range (0, len(output)):
        for x in range(0, win_step):

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
    return np.nonzero(relu_flat_output)[0]


def format_classifier_input(spike_indexes, data_sequence, win_size):
    """Returns only the formatted input_data to be fed into the classification model to make predictions

    Params:
    - spike_indexes: Array, containing the predicted indexes of the spikes in the neuron recording
    - data_sequence: Array, the time domain data of the neuron recording
    - win_size: Integer, the length of the windows the input_data will be formatted to
    """

    half_win_size = win_size // 2
    sequence_len = len(data_sequence)

    input_data = []

    for i in spike_indexes:
        # max and min used to prevent the index of the data_sequence from being exceeded
        d_window = data_sequence[max(0, (i - half_win_size)): min(sequence_len, (i + half_win_size))]

        if len(d_window) != win_size:
            # Padding containing the average value of the window is added before and after to make the window meet the win_size
            avg = np.mean(d_window)
            upper_padding = half_win_size - (min(sequence_len, (i + half_win_size)) - max(0, i))
            lower_padding = half_win_size - (min(sequence_len, i) - max(0, (i - half_win_size)))

            d_window = np.concatenate([[avg] * lower_padding, d_window, [avg] * upper_padding], axis=0)

        input_data.append(d_window)

    input_data = np.array(input_data).reshape(-1, win_size)

    return input_data


def format_classifier_training_data(
        spike_indexes, class_sequence, 
        data_sequence, win_size):
    """Returns the formatted input_data and label_data to be fed into the classification model for training

    Params:
    - spike_indexes: Array, containing the predicted indexes of the spikes in the neuron recording
    - class_sequence: Array, containing the class value of the spike at the index where each spike is recorded
    - data_sequence: Array, the time domain data of the neuron recording
    - win_size: Integer, the length of the windows the input_data will be formatted to
    """
    
    num_of_classes = 5

    half_win_size = win_size // 2
    sequence_len = len(data_sequence)

    train_data = []
    label_data = []

    for i in spike_indexes:
        for x in range(-6, 6): # Jitter used to increase training data and train for inaccurate spike predictions

            # max and min used to prevent the index of the data_sequence or class_sequence from being exceeded
            d_window = data_sequence[max(0, (i + x - half_win_size)): min(sequence_len, (i + x + half_win_size))] 
            upper_half = class_sequence[max(0, (i + x)): min(sequence_len, (i + x + half_win_size))]
            bottom_half = class_sequence[max(0, (i + x - half_win_size)): min(sequence_len, (i + x))]

            if len(d_window) != win_size:
                # Padding containing the average value of the window is added before and after to make the window meet the win_size
                avg = np.mean(d_window)
                upper_padding = half_win_size - len(upper_half)
                lower_padding = half_win_size - len(bottom_half)

                d_window = np.concatenate([[avg] * lower_padding, d_window, [avg] * upper_padding], axis=0)

            train_data.append(d_window)

            # Gets a list of the indexes of the non-zero values of each half of the class_sequence window
            upper_class_index = np.nonzero(upper_half)[0]
            bottom_class_index = np.nonzero(bottom_half)[0]
            
            label_options = [0] * num_of_classes

            # Sets nearest_class to the class value closest to the corresponding predicted spike index
            if (len(upper_class_index) == 0) and (len(bottom_class_index) == 0):
                # If no non-zero values are found in either half then no class should be predicted
                label_data.append(label_options)
                break
            elif len(upper_class_index) == 0:
                nearest_class = int(bottom_half[bottom_class_index[0]])
            elif len(bottom_class_index) == 0:
                nearest_class = int(upper_half[upper_class_index[0]])
            elif upper_class_index[0] <= ((win_size//2) - bottom_class_index[0]):
                nearest_class = int(upper_half[upper_class_index[0]])
            else:
                nearest_class = int(bottom_half[bottom_class_index[0]])
        
            label_options[nearest_class -1] = 1
            label_data.append(label_options)
    
    train_data = np.array(train_data).reshape(-1, win_size)
    label_data = np.array(label_data).reshape(-1, num_of_classes)

    return train_data, label_data


def get_classifier_training_data(pred_spike_indexes, win_size):
    """Returns the train_data, label_data, val_train_data, val_label_data that will be passed to the classification model to train

    Params:
    - pred_spike_indexes: Array, containing the indexes of the predicted spikes
    - win_size: Integer, the length of the window that the training data is formatted to fit into
    """
    
    d1_noisy, Index, Class = load_dataset("D1.mat", labelled=True, original=True)
    d1_filtered = load_dataset("D1_Noisy_filtered.mat", labelled=False, original=False)

    sequence_len = len(d1_filtered)

    class_sequence = create_class_sequence(Index, Class, sequence_len)

    training_spike_indexes = pred_spike_indexes[0:int(len(pred_spike_indexes) * 0.8)]
    validation_spike_indexes = pred_spike_indexes[int(len(pred_spike_indexes) * 0.8):]
    train_data, label_data = format_classifier_training_data(training_spike_indexes, class_sequence, d1_filtered, win_size)
    val_train_data, val_label_data = format_classifier_training_data(validation_spike_indexes, class_sequence, d1_filtered, win_size)

    return train_data, label_data, val_train_data, val_label_data

            
def classifier_post_processing(classifier_output):
    """Returns a list with the class value predicted by the classifier model

    Params:
    - classifier_output: Array, the output of the classification model, split into windows
    """
    
    predicted_classes = []

    for i in classifier_output:
        highest_prob_class = int(np.argmax(i)) + 1  # Gets the index of the class it is most likely to be
        predicted_classes.append(highest_prob_class)
    
    return predicted_classes

