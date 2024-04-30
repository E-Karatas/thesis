clear variables; home; close all

% Specify the Excel file path
excelFilePath = 'bio_alloys\BioAlloysValidationForMatlab.xlsx';
% Read data from Excel file into a table
dataTable = readtable(excelFilePath);

% Initialize the table with zeros
bio_Ti_df = array2table(zeros(height(dataTable), 2));
% Assign variable names as a cell array
bio_Ti_df.Properties.VariableNames = {'alloy','alloyType'};
bio_Ti_df.alloy = dataTable.alloy;
bio_Ti_df.alloyType = dataTable.alloyType;

% Remove spaces from the 'alloy' variable
bio_Ti_df.alloy = strrep(bio_Ti_df.alloy, ' ', '');
% Ensure that the delimiters are correct
bio_Ti_df.alloy = strrep(bio_Ti_df.alloy, '–', '-');
% Use unique function to get unique rows
uniqueRows = unique(bio_Ti_df, 'stable');

% Rename the table
bio_Ti_df = uniqueRows;

% Load the table containing the Ti alloys from the article with the
% deformation mech label
load('data4/df_all.mat');

% Check for elements in bio_Ti_df.alloy that occur in df2.alloy
isMemberIdx = ismember(bio_Ti_df.alloy, df2.alloy);

% Eliminate elements from bio_Ti_df.alloy that occur in df2.alloy
bio_Ti_df = bio_Ti_df(~isMemberIdx, :);

%Get compositions in molar fractions
for ii = 1:height(bio_Ti_df)
    composition{ii} = get_composition(bio_Ti_df.alloy{ii});
end
%Append compositions to table
bio_Ti_df.composition = composition';

