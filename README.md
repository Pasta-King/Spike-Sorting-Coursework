# EE32009-CW

## Dependancies:
- numpy
- scipy
- keras
- tensorflow

## How to run:
Ensure that the necessary datasets are included in the folder Coursework-Datasets/

Run Matlab_Filter.m changing dataset variable to the dataset you want to filter, a Butterworth filter will be applied producing the array re_wave. 
Then open the Wavelet Signal Denoiser app from the App Selector.
Import re_wave into the Wavelet Signal Denoiser by clicking import and then selecting re_wave from the workspace.
Once the denoiser has finished, export it's results by clicking Export

Uncomment the lines
```
% hold on
% plot(re_wave1, "r")

% save(output_dataset, "re_wave1")
```

Then run the Matlab_Filter.m again, this should produce a filtered dataset under the Filtered_Datasets/ folder

Then run TrainingDetection.py
Then run TrainingClassifier.py
Then run PredictingResults.py

And now the results/ folder should contain the .mat files of the results of the models. 
This can be zipped and submitted to the automarker.

## AI Disclosure
I acknowledge that this work is my own, and I have used Copilot (Microsoft, https://copilot.microsoft.com/) to suggest an outline for solutions only
