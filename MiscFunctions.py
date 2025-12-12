import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import keras
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import butter, filtfilt
import pywt

def filter_wave(wave):
    order = 5
    threshold = 5
    sample_freq = 25e3

    # gaussian_filter1d(wave, order)
    normal_threshold = threshold / (0.5 * sample_freq)
    b, a = butter(order, normal_threshold, btype="high", analog=False)
    return filtfilt(b, a, wave)

def test_wavelet_coeffs(wave):
    coeffs = pywt.wavedec(wave, "sym4", level=17)
    approx_coeffs = coeffs[0]
    detail_coeffs = coeffs[1:]

    re_wave = pywt.waverec([approx_coeffs] + detail_coeffs, "sym4")

    return re_wave

def nb_beta(sigma, L, detail):
    S2 = np.sum(detail ** 2)
    lmbd = 4.50524
    beta = (1 - lmbd * L * sigma**2 / S2)
    return max(0, beta)

def neigh_block(details, n, sigma):
    res = []
    L0 = int(np.log2(n) // 2)
    L1 = max(1, L0 //2)
    L = L0 + 2 * L1

    for d in details:
        d2 = d.copy()
        for start_b in range(0, len(d2), L0):
            end_b = min(len(d2), start_b + L0)
            start_B = start_b - L1
            end_B = start_B + L
            if start_B < 0:
                end_B -= start_B
                start_B = 0
            elif end_B > len(d2):
                start_B -= end_B - len(d2)
                end_B = len(d2)
            assert end_B - start_B == L
            d2[start_b:end_b] *= nb_beta(sigma, L, d2[start_B:end_B])
        res.append(d2)
    return res

def test_nb_filter(wave):
    coeffs = pywt.wavedec(wave, "sym4")
    approx_coeffs = coeffs[0]
    detail_coeffs = coeffs[1:]

    details_nb = neigh_block(detail_coeffs, len(wave), 0.8)

    wave_nb = pywt.waverec([approx_coeffs] + details_nb, "sym4")

    return wave_nb

def autocorr(x):
    result = np.correlate(x, x, mode="full")
    return result[int(len(result) / 2) :]

def test(corr: np.ndarray):
    thr = 1.96 / np.sqrt(len(corr))
    tests = np.where(np.abs(corr) <= thr, 0, 1)
    return tests.sum() == 0

def test_wavelet_filter(wave, threshold):
    coeffs = pywt.wavedec(wave, "sym4")

    coeffs_thresh = [pywt.threshold(c, threshold, "soft") for c in coeffs]
    re_wave = pywt.waverec(coeffs_thresh, "sym4")

    return re_wave

def find_wavelet_threshold(wave):
    threshold = 0
    step = 0.0001

    for i in range(1000):
        threshold = step * i
        re_wave = test_wavelet_filter(wave, threshold)
        err = re_wave - wave
        corr = autocorr(err)

        if not test(corr):
            threshold = threshold - step
            print("Optimal threshold: ", threshold)
            break

    return threshold

def extend_wave():
    wave, Indexes, Classes = load_dataset("D1.mat", labelled=True, original=True)

    sequence_len = len(wave)

    print(len(wave))
    print(Indexes[-1])

    empty_Indexes = np.empty(0, dtype="double")
    empty_Classes = np.empty(0, dtype="double")
    for i in range(0, 4):
        new_Index =   [int(x + sequence_len * i) for x in Indexes] # Indexes + int(sequence_len * i) #
        new_Index = np.array(new_Index, dtype=np.int32)
        new_Index.astype(np.double)
        empty_Indexes = np.concatenate([empty_Indexes, new_Index])
        empty_Classes = np.concatenate([empty_Classes, Classes])

    wave = np.concatenate([wave, wave, wave, wave])

    print(len(wave))
    print(empty_Indexes[-1])

    noisy_wave = add_noise(wave, Indexes)

    m_dict = {"d": noisy_wave, "Index": empty_Indexes, "Class": empty_Classes}
    spio.savemat("Noisy_Datasets/D1_NosyAndLong_v3.mat", mdict=m_dict)