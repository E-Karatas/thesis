clear variables; home; close all

load('data3/df_microstructure.mat');

df2 = df;

% Initialize an empty matrix to store the content columns
TempDF = zeros(height(df2), height(df2.composition{1,1}.content));

% Access the row names from the first composition element to initialize the columnNames cell array
firstCompositionTable = df2.composition{1, 1};
columnNames = cellstr(firstCompositionTable.Properties.RowNames);

% Loop through each row in the df2 table
for ii = 1:height(df2)
    % Access the 'composition' table
    compositionTable = df2.composition{ii, 1};
    
    % Transpose and store the content variable in the matrix
    TempDF(ii, :) = transpose(compositionTable.content);
end

% Convert the matrix to a table
TempDFTable = array2table(TempDF);

% Set the column names for the table
TempDFTable.Properties.VariableNames = columnNames;

% Find the indices of columns where every element is 0
zeroColumns = all(TempDFTable{:,:} == 0, 1);

% Remove the zero columns from the table
TempDFTable(:, zeroColumns) = [];

% Append TempDFTable to df2
df2 = [df2, TempDFTable];