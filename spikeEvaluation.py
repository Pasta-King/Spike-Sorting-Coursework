import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras

model_version = 2

mat = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]

sequence_len = len(d[0])
train_start = 0
train_end = int(sequence_len * 0.8)

win_size = 200
input_shape = (win_size, 1)
win_step = 160

d_input = []
for i in range(train_start, sequence_len - win_size, win_step):
    d_input.append(d[0][i:i + win_size])

d_input = np.array(d_input).reshape(-1, win_size)

model = keras.models.load_model("models/spike_detection_v" + str(model_version) + ".keras")

output = model.predict(d_input)

# print("full output")
# print(output)
# print("output[0]")
# print(output[0])
# print("output[0][0]")
# print(output[0][0])
# print("output[0][0][0]")
# print(output[0][0][1])

print(Index)

x = np.arange(0, sequence_len).tolist()
plt.plot(x, d[0])


relu_flat_output = []
error_spikes = 0

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

#print(len(output))


total_true_spikes = 0
total_in_range_spikes = 0
total_fake_spikes = 0

# Going through predicted spikes and checking accuracy
predicted_spikes = np.nonzero(relu_flat_output)[0]

for i in range(0, len(Index[0])):
    spike_index = Index[0][i]
    
    plt.plot(spike_index, d[0][spike_index], "kx")

for i in predicted_spikes:
    true_spike = 0
    in_range_spike = 0

    for x in Index[0]:
        if i == x:
            plt.vlines(i, 0, 4, colors="g")
            total_true_spikes += 1
            true_spike = 1
    
    if true_spike == 0:
        for x in Index[0]:
            if (i < x + 50) and (i > x - 50) and (i != x) and (true_spike == 0):
                plt.vlines(i, 0, 4, colors="m")
                total_in_range_spikes += 1
                in_range_spike = 1
    
    if (true_spike == 0) and (in_range_spike == 0):
        plt.vlines(i, 0, 4, colors="r")
        total_fake_spikes += 1


# Going through real spikes and checking accuracy]
# for i in range(0, len(Index[0])):
#     spike_index = Index[0][i]
    
#     plt.plot(spike_index, d[0][spike_index], "kx")

#     if (len(relu_flat_output) > spike_index + 50) and (spike_index > 50):
#         accurate_window = relu_flat_output[Index[0][i] - 50 : Index[0][i] + 50]
#     elif len(relu_flat_output) > spike_index + 50:
#         accurate_window = relu_flat_output[: Index[0][i] + 50]
#     elif spike_index > 50:
#         accurate_window = relu_flat_output[Index[0][i] - 50 :]
#     else:
#         accurate_window = relu_flat_output[:]

#     predicted_spikes = np.nonzero(accurate_window)[0]

#     if relu_flat_output[spike_index] == 1:
#         total_true_spikes += 1
#         plt.vlines(spike_index, 0, 4, colors="g")

#     if len(predicted_spikes) == 1:
#         plt.plot(spike_index, d[0][spike_index], "gx")
#     elif len(predicted_spikes) > 1:
#         plt.plot(spike_index, d[0][spike_index], "gx")
    


    

print("total spikes: ", len(Index[0]))
print("total predicted spikes:", len(predicted_spikes))
print("true spikes: ", total_true_spikes)
print("spikes in range: ", total_in_range_spikes)
print("missing spikes: ", len(Index[0]) - (total_true_spikes + total_in_range_spikes))
print("error spikes: ", error_spikes)
print("fake spikes: ", total_fake_spikes)

plt.show()

    # for x in np.nonzero(output[i][:win_step] > 0.25)[0]:
    #     spikes.append(int(x) + i*160)

# for i in Index:
#     true_spike = np.nonzero(spikes == Index)
#     spike_in_range = np.nonzero(spikes > Index - 50 & spikes < Index + 50)

# print("Predicted Spikes", len(spikes))
# spikes = np.flatnonzero(output > 0.25)
# print(spikes)
# print("Predicted Spikes", len(spikes))
# print(np.sort(Index)[0])
# print("Real Spikes", len(np.sort(Index)[0]))