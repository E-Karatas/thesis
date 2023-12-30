clear variables; home; close all

load('data/alloy_data7.mat');

logicalIndex = cellfun(@isempty, T.Deformation_mechanism_simple);

% Create a new table with rows where Deformation_mechanism_simple is not empty
df3 = T(logicalIndex, :);

% Specify the columns to be removed
columnsToRemove = {'Ms', 'Deformation_mechanism'};

% Remove specified columns and create a new table
df4 = removevars(df3, columnsToRemove);