clear variables; home; close all

load('T0_data/True_label_data.mat');

% Initialize an empty cell array to store the content columns
TempDF = zeros(height(df2), 112);

% this is how you can access the content column in the composition table 
% df2.composition{1,1}.content
% 
for ii = 1:height(df2)
    TempDF(ii,:) = transpose(df2.composition{ii,1}.content);
end

