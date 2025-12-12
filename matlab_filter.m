%{
Matlab_Filter.py contains the sequence used to produce filtered versions of the dataset

The variable dataset is set to the name of the dataset to be filtered.
A Butterworth high pass filter is used to remove the low frequency noise
from the neuron recording producing the array re_wave. 
Then the Wavelet Signal Denoiser needs to be selected from the App
Selector.
The array re_wave needs to be imported into the Wavelet Signal Denoiser.
The output of the Wavelet Signal Denoiser then needs to be exported back to
the Workspace.

Then the lines:
> % hold on
> % plot(re_wave1, "r")
>
> % save(output_dataset, "re_wave1")

Need to be uncommented and Matlab_Filter.m must be run again

%}

dataset = "D2";

input_dataset = "Coursework-Datasets\" + dataset + ".mat";
output_dataset = "Filtered_Datasets\"+ dataset + ".mat";

load(input_dataset, "d")

normal_threshold = 5 / (0.5 * 25000);
[b, a] = butter(5, normal_threshold, "high");
re_wave = filtfilt(b, a, d);

plot(d, "b")

% hold on
% plot(re_wave1, "r")

% save(output_dataset, "re_wave1")