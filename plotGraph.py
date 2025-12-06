import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d


def butter_highpass_filter(signal, threshold, freq, order=5):
    normal_threshold = threshold / (0.5 * freq)
    b, a = butter(order, normal_threshold, btype="high", analog=False)
    return filtfilt(b, a, signal)

mat_d6 = spio.loadmat("Coursework-Datasets-20251028/D6.mat")
d6 = mat_d6["d"]

mat_d1 = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d1 = mat_d1["d"]
Index = mat_d1["Index"]
# Class = mat["Class"]
win_start = 0
win_end = len(d6[0])
win_size = win_end - win_start


d1_x = np.linspace(win_start, win_end, win_size, dtype=int)
d6_x = np.linspace(0, len(d6[0]), len(d6[0]), dtype=int)

# Adding Noise
noise = np.random.normal(0, 2, [win_size]) 
noise_wave = 3 * np.sin(2*np.pi*0.00000286*d1_x + 4) #0.00001
d1_noisy = d1[0]* 2 + noise + noise_wave - 5

#Filtering
filtered_mat = spio.loadmat("Filtered_Datasets/D1_filtered.mat")
d1_filtered = filtered_mat["re_wave1"]
d1_filtered = d1_filtered[:, 0]
# d1_filtered = gaussian_filter1d(d1_noisy, 5)
# d1_filtered = butter_highpass_filter(d1_filtered, 5, 25e3)
# d6_filtered = gaussian_filter1d(d6[0], 5)
# d6_filtered = butter_highpass_filter(d6_filtered, 5, 25e3)

#filtered_d6 = gaussian_filter(d6[0], 3)

#print(s_Index[:10])
sorted_Index = np.sort(Index[0])
min_diff = len(sorted_Index)
total_short_diffs = 0

lowest_spike_amplitude = 25e5
sum_of_spike_amplitudes = 0
amplitudes_less_than_zero = 0

print(sorted_Index)
for i in range(0, len(sorted_Index)-1):
    diff = sorted_Index[i+1] - sorted_Index[i]
    if diff < 10:
        total_short_diffs += 1
    if min_diff > diff:
        print(sorted_Index[i+1], " - ", sorted_Index[i])
        min_diff = diff
        print("min_diff = ", min_diff)

    amplitude = d1_filtered[sorted_Index[i]]

    sum_of_spike_amplitudes += amplitude 

    if amplitude < lowest_spike_amplitude:
        lowest_spike_amplitude = amplitude
        print("min_amp = ", lowest_spike_amplitude)

    if amplitude <= 0 :
        amplitudes_less_than_zero += 1
    

print("spikes less than 10 away: ", total_short_diffs)
print("lowest amplitude", lowest_spike_amplitude)
print("average amplitude", str(sum_of_spike_amplitudes / len(sorted_Index)))
print("amp less than 0", amplitudes_less_than_zero)

# plt.plot(d6_x, d6[0], "b")
# plt.plot(d6_x, d6_filtered, "r")
plt.plot(d1_x, d1[0], "g")
plt.plot(d6_x, d1_filtered, "r")
# plt.plot(s_Index[0], d_sample[s_Index[0] - win_start], "rx")
# plt.plot(s_Index[1], d_sample[s_Index[1] - win_start], "rx")
plt.show()