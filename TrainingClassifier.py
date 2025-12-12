"""TrainingClassifier.py contains the sequence used to create and train the classification CNN model

Sets the input window size and the steps between each input window. Creates the classification CNN model by adding 
different convolution and dense layers. And then fits the training data to the model using the Categorical Cross 
Entropy loss function. Saves the model in the models/ folder.

Dependencies:
- keras
"""

from keras import layers, models, losses
from DataProcessing import get_full_detection_data, detection_post_processing, get_classifier_training_data



# Version of the classification model being created
new_classifier_version = 1
# Version of the detection model used to predict spike indexes
detection_version = 1

detect_win_size = 50
detect_win_step = 30

detection_input_data, detection_label_data = get_full_detection_data(detect_win_size, detect_win_step)

detector_model = models.load_model("models/spike_detection_v" + str(detection_version) + ".keras")

print("Detecting spike indexes")
output = detector_model.predict(detection_input_data)

predicted_spikes = detection_post_processing(output, detect_win_step)

classifier_win_size = 100
input_shape = (classifier_win_size, 1)

train_data, label_data, val_train_data, val_label_data = get_classifier_training_data(predicted_spikes, classifier_win_size)


model = models.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Normalization(axis=None))
model.add(layers.MaxPooling1D(4))
model.add(layers.Conv1D(20, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(50, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(150, 3, padding="same", activation="sigmoid"))
model.add(layers.Flatten())
model.add(layers.Dense(80, activation="sigmoid"))
model.add(layers.Dense(30, activation="sigmoid"))
model.add(layers.Dense(5, activation="sigmoid"))
model.summary()

model.compile(optimizer='adamW', loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])

print("Training Classifier")
history = model.fit(train_data, label_data, epochs=120, batch_size=18, validation_data=(val_train_data, val_label_data)) 

# Model is saved in the models folder
model.save("models/spike_classification_v" + str(new_classifier_version) + ".keras")