import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

win_start = 1300
win_end = 2000
win_size = win_end - win_start

mat = spio.loadmat("Coursework-Datasets-20251028/D1.mat")
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]

d_sample = d[0][win_start:win_end]
x = np.linspace(win_start, win_end, win_size, dtype=int)

s_Index = np.sort(Index[0])

print(s_Index[:10])
plt.plot(x, d_sample)
plt.plot(s_Index[0], d_sample[s_Index[0] - win_start], "rx")
plt.plot(s_Index[1], d_sample[s_Index[1] - win_start], "rx")
plt.show()