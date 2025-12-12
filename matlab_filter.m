
% load("Coursework-Datasets-20251028\D1.mat", "d", "Index")

% load("D1_Noisy.mat", "d")

disp(d(end));
disp(Index(end));


normal_threshold = 5 / (0.5 * 25000);
[b, a] = butter(5, normal_threshold, "high");
re_wave = filtfilt(b, a, d);

plot(d, "b")
hold on
plot(re_wave1, "r")

save("Filtered_Datasets\D1_LongAndFiltered_v2", "re_wave1")