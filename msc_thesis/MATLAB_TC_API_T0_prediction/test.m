% Implementation of 
% M. Bignon, E. Bertrand, F. Tancret, P.E.J. Rivera-díaz-del-castillo, 
% Modelling martensitic transformation in titanium alloys : 
% the influence of temperature and deformation, Materialia. 7 
% (2019) 100382. https://doi.org/10.1016/j.mtla.2019.100382.
clear variables; home; close all

%% Import alloy information and extract data
load('data/Labeled_data.mat');

T = df3;

%Get compositions in molar fractions
for ii = 1:height(T)
    composition{ii} = get_composition(T.alloy{ii});
end
%Append compositions to table
T.composition = composition';

