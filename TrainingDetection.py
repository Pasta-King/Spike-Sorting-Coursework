"""TrainingDetection.py contains the sequence used to create and train the detection CNN model

Sets the input window size and the steps between each input window. Creates the detection CNN model by adding 
different convolution and dense layers. And then fits the training data to the model using the Binary Cross Entropy
loss function. Saves the model in the models/ folder.

Dependencies:
- keras
"""

from keras import layers, models, losses
from DataProcessing import get_detection_training_data


# Version of the detection model being created
new_detection_version = 1

win_size = 50
win_step = 30

train_data, label_data, val_train_data, val_label_data = get_detection_training_data(win_size, win_step)


input_shape = (win_size, 1)
model = models.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Normalization(axis=None))
model.add(layers.Conv1D(34, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(64, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(120, 3, padding="same", activation="sigmoid"))
model.add(layers.Conv1D(256, 3, padding="same", activation="sigmoid"))
model.add(layers.Dense(800, activation="sigmoid"))
model.add(layers.Dense(600, activation="sigmoid"))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer='adamW', loss=losses.BinaryCrossentropy(), metrics=["accuracy"])

# Model is trained on the training data
history = model.fit(train_data, label_data, epochs=30, batch_size=18, validation_data=(val_train_data, val_label_data)) 

# Model is saved in the models/ folder
model.save("models/spike_detection_v" + str(new_detection_version) + ".keras")

