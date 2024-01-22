clear variables; home; close all

load('data4/df_all2.mat');

% Convert empty arrays, [], to empty strings to keep the data type
% consistent this is not a problem for the bio dataframe because it is not
% a datatype there
microstructureColumn = df2.Microstructure;
dms_Column = df2.Deformation_mechanism_simple;
dm_Column = df2.Deformation_mechanism;

microstructureColumn(cellfun(@isempty, microstructureColumn)) = {''};
dms_Column(cellfun(@isempty, dms_Column)) = {''};
dm_Column(cellfun(@isempty, dm_Column)) = {''};

% Replace the column in the original table
df2.Microstructure = microstructureColumn;
df2.Deformation_mechanism_simple = dms_Column;
df2.Deformation_mechanism = dm_Column;

% Save the table to Parquet format
parquetwrite('df_all.parquet', df2);