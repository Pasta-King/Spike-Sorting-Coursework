load("Noisy_Datasets\D1_Noisy.mat", "d")

normal_threshold = 5 / (0.5 * 25000);
[b, a] = butter(5, normal_threshold, "high");
re_wave = filtfilt(b, a, d);

plot(d, "b")
hold on
plot(re_wave1, "r")

save("Filtered_Datasets\D1_Noisy_filtered.mat", "re_wave1")