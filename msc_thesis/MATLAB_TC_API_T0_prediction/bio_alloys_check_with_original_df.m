clear variables; home; close all

load('bio_alloys/bio_alloys_df.mat');
load('data4\df_all.mat')

% Use unique function to get unique rows
uniqueRows = unique(bio_alloys_df, 'stable');

% Remove spaces from the 'alloy' variable
uniqueRows.alloy = strrep(uniqueRows.alloy, ' ', '');

% Rename the table
% bio_Ti_df = uniqueRows;