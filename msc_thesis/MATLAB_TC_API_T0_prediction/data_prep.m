clear variables; home; close all

load('data/alloy_data7.mat');

% Assuming your original table is named 'T'
% Replace this with the actual variable name you have in your workspace

% Create a logical index for rows where Deformation_mechanism_simple is not empty
logicalIndex = ~cellfun(@isempty, T.Deformation_mechanism_simple);

% Create a new table with rows where Deformation_mechanism_simple is not empty
df = T(logicalIndex, :);

% Assuming your original table is named 'T'
% Replace this with the actual variable name you have in your workspace

% Specify the columns to be removed
columnsToRemove = {'Ms', 'Deformation_mechanism'};

% Remove specified columns and create a new table
df2 = removevars(df, columnsToRemove);