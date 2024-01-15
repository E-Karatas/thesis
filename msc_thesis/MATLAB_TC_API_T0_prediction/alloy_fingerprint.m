clear variables; home; close all

load('bio_alloys/bio_Ti_alloys.mat');

% remove Mn, Si, Zn elements because they aren't a feature in our original dataset
% Identify rows to keep
rowsToKeep = ~contains(bio_Ti_df.alloy, 'Mn') & ~contains(bio_Ti_df.alloy, 'Si') & ~contains(bio_Ti_df.alloy, 'Zn');
% Filter the table to keep only the desired rows
bio_Ti_df = bio_Ti_df(rowsToKeep, :);

%Get compositions in molar fractions
for ii = 1:height(bio_Ti_df)
    composition{ii} = get_composition(bio_Ti_df.alloy{ii});
end
%Append compositions to table
bio_Ti_df.composition = composition';

df2 = bio_Ti_df;

% Initialize an empty matrix to store the content column
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

% Rename df2 to bio_alloys_df
bio_alloys_df = df2;

% Remove the 'composition' column
bio_alloys_df.composition = [];
