import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio



mat_d6 = spio.loadmat("Coursework-Datasets-20251028/D6.mat")
d6 = mat_d6["d"]
mat_d1 = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d1 = mat_d1["d"]
# Index = mat["Index"]
# Class = mat["Class"]
win_start = 0
win_end = len(d6[0])
win_size = win_end - win_start

noise = np.random.normal(0, 1, [win_size]) 


d1_noisy = d1[0][win_start:win_end] + noise
d1_x = np.linspace(win_start, win_end, win_size, dtype=int)
d6_x = np.linspace(0, len(d6[0]), len(d6[0]), dtype=int)

noise_wave = 15 * np.sin(2*np.pi*0.5*d1_x)
# s_Index = np.sort(Index[0])

#print(s_Index[:10])
plt.plot(d6_x, d6[0], "b")
plt.plot(d1_x, d1_noisy, "r")
plt.plot(d1_x, noise_wave, "g")
# plt.plot(s_Index[0], d_sample[s_Index[0] - win_start], "rx")
# plt.plot(s_Index[1], d_sample[s_Index[1] - win_start], "rx")
plt.show()