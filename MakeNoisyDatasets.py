"""MakeNoisyDatasets.py contains the sequence used to create additional datasets based on D1 with different amounts of simulated noise

Loads the original D1 dataset. Passes the neuron recording and Index values to the add_noise function which is configured to add noise 
to make D1 appear similar to D6. And then saves the new wave in the Noisy_Datasets/ folder.

Dependencies:
- numpy
- scipy
"""

import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from DataProcessing import load_dataset, save_wave



def add_noise(wave, spikes):
    """Returns a wave with added noise

    Params:
    - wave: Array, containing the original wave without the added noise
    = spikes: Array, containing the indexes of the spikes present in the wave
    """

    target_signal_noise_ratio = -5
    lowband_threshold = 0.5
    low_broad_ratio = 0.8
    sampling_freq = 25e3
    rng = np.random.default_rng(1)
    sequence_size = wave.size
    avg_power = np.mean(wave ** 2) + 1e-20

    rfft_array = np.zeros(rfft(wave).shape, dtype=complex)

    freq_bin_centres = rfftfreq(sequence_size, d=(1.0 / sampling_freq))
    lowband_mask = freq_bin_centres <= lowband_threshold
    rfft_lowband = (rng.normal(size=lowband_mask.sum()) + 1j * rng.normal(size=lowband_mask.sum()))
    rfft_array[lowband_mask] = rfft_lowband
    low_freq_noise = irfft(rfft_array, n=sequence_size)

    broad_freq_noise = rng.normal(size=sequence_size)

    low_freq_noise = low_freq_noise / (np.sqrt(np.mean(low_freq_noise ** 2)) + 1e-20)
    broad_freq_noise = broad_freq_noise / (np.sqrt(np.mean(broad_freq_noise ** 2)) + 1e-20)
    total_noise = (low_broad_ratio * low_freq_noise + (1 - low_broad_ratio) * broad_freq_noise)
    total_noise = total_noise / (np.sqrt(np.mean(total_noise ** 2)) + 1e-20)

    required_noise_power = avg_power / (10.0 ** (target_signal_noise_ratio / 10.0))
    noise = total_noise * np.sqrt(required_noise_power)

    spike_mask = np.zeros(sequence_size, dtype=bool)
    for i in spikes:
        lb = max(0, i)
        ub = min(sequence_size, i + 30)
        spike_mask[lb:ub] = True
    noise[spike_mask] *= 0.1

    return wave + noise


def extend_wave():
    """Saves a new dataset that is four times the length of the original with added noise"""
    
    wave, Indexes, Classes = load_dataset("D1.mat", labelled=True, original=True)

    sequence_len = len(wave)

    extended_Indexes = np.empty(0, dtype="double")
    extended_Classes = np.empty(0, dtype="double")
    for i in range(0, 4):
        new_Index =   [int(x + (sequence_len * i)) for x in Indexes]
        new_Index = np.array(new_Index, dtype=np.int32)
        new_Index.astype(np.double)
        extended_Indexes = np.concatenate([extended_Indexes, new_Index])
        extended_Classes = np.concatenate([extended_Classes, Classes])

    wave = np.concatenate([wave, wave, wave, wave])

    noisy_wave = add_noise(wave, Indexes)

    save_wave(d1_noisy, "Noisy_Datasets/D1_NosyAndLong_v3.mat", indexes=extended_Indexes, classes=extended_Classes)



if __name__ == "__main__":
    d1, Index, Class = load_dataset("D1.mat", labelled=True, original=True)

    d1_noisy = add_noise(d1, Index)

    save_wave(d1_noisy, "Noisy_Datasets/D1_Noisy.mat")

    # Now use matlab_filter.m to create the Filtered Datasets