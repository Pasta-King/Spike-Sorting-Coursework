import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras

model_version = 2

mat = spio.loadmat("Coursework-Datasets-20251028/D5.mat")
d = mat["d"]

sequence_len = len(d[0])
train_start = 0
train_end = int(sequence_len * 0.8)

win_size = 200
input_shape = (win_size, 1)
win_step = 160

d_train = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_train.append(d[0][i:i + win_size])

d_train = np.array(d_train).reshape(-1, win_size)

model = keras.models.load_model("models/spike_detection_v" + str(model_version) + ".keras")

output = model.predict(d_train)

x = np.arange(0, sequence_len).tolist()
plt.plot(x, d[0])


relu_flat_output = []

# For BinaryCrossEntropy
#for i in range (0, len(output)):
#    relu_layer = [1 if x > 0.5 else 0 for x in output[i][:160]]
#    relu_flat_output = relu_flat_output + relu_layer

# For SparseCategoricalCrossEntropy
for i in range (0, len(output)):
    for x in range(0, win_step):
        if output[i][x][1] >  output[i][x][0]:
            relu_flat_output.append(1)
        elif output[i][x][0] >=  output[i][x][1]:
            relu_flat_output.append(0)
        else:
            print("Error spike at: ", len(relu_flat_output))
            error_spikes += 1
            relu_flat_output.append(-1)



# Going through predicted spikes and checking accuracy
predicted_spikes = np.nonzero(relu_flat_output)[0]

for i in predicted_spikes:

    plt.vlines(i, 0, 4, colors="g")


plt.show()
