
load("Coursework-Datasets-20251028\D1.mat", "d", "Index")

target_signal_noise_ratio = -5;
lowband_threshold = 0.5;
low_broad_ratio = 0.8;
sampling_freq = 25e3;
rng("default");
sequence_size = length(d);
disp(sequence_size)
avg_power = mean(d .^ 2) + 1e-20;

d_fft = fft(d);
rfft_array = zeros(size(fft(d))); %, typename="COMPLEX_DOUBLE"
 %disp(size(fft(d)));

freq_bin_centres = fftfreq(sequence_size, d=(1.0 / sampling_freq));


normal_threshold = 5 / (0.5 * 25000);
[b, a] = butter(5, normal_threshold, "high");
re_wave = filtfilt(b, a, d);

plot(d, "b")
% hold on
% plot(re_wave1, "r")

% save("Filtered_Datasets\D1_LongAndFiltered_v2", "re_wave1")