clear variables; home; close all

load('bio_alloys/bio_alloys_df.mat');
load('data4/df_all.mat');

% Add the missing elements to the filtered dataframe of biocompatible Ti alloys. 
% The missing elements are Al, Cr, W, V
bio_alloys_df.Al = zeros(height(bio_alloys_df), 1);
bio_alloys_df.Cr = zeros(height(bio_alloys_df), 1);
bio_alloys_df.W = zeros(height(bio_alloys_df), 1);
bio_alloys_df.V = zeros(height(bio_alloys_df), 1);

% Check if the bio_alloys_df have elements that are already in our df
% Assuming bio_alloys_df and df2 are tables with Al, Cr, Fe, Mo, Nb, O, Ta, Sn, Ti, W, V, Zr columns
% Define the list of elements
elements = {'Al', 'Cr', 'Fe', 'Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr'};

% Create a logical mask for rows in bio_alloys_df that match any row in df2 across all elements
matchingRows = all(ismember(bio_alloys_df{:, elements}, df2{:, elements}), 2);

% Remove matching rows from bio_alloys_df
bio_alloys_df(matchingRows, :) = [];
