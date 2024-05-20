Master thesis - Computational design of advanced Titanium alloys using physics-informed machine learning.

Most of the data preparation and preprocessing was done in matlab. The MATLAB.. folder contains some of the preprocessing scripts including the scripts used to calculate the T0 temperature with ThermoCalc (courtesy Frank Niessen).

The training and test set can be found in the data folder with the names df_matminer.pkl (training set) and validation_set_matminer.pkl (this is actually the test set and not the validation set). The other parquet and pickle files were used in earlier iterations of the project and will be removed soon as they're redundant.

The linear- and random forest regression models trained to predict the T0 temperature can be found in the T0_prediction folder.

The logistic regression (classification) and random forest classification models trained to predict the deformation mechanism of our alloys can be found in the Behaviour_prediction folder.
