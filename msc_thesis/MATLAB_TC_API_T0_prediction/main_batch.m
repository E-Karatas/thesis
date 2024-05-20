% Implementation of 
% M. Bignon, E. Bertrand, F. Tancret, P.E.J. Rivera-díaz-del-castillo, 
% Modelling martensitic transformation in titanium alloys : 
% the influence of temperature and deformation, Materialia. 7 
% (2019) 100382. https://doi.org/10.1016/j.mtla.2019.100382.
clear variables; home; close all

%% Initialization of variables
temp_ini = 1000; % Initial temperature for Ms_star search
%Thermo-Calc settings
TC.database = "TCTI4";
TC.parent_phase = "BCC_B2";
TC.child_phase = "HCP_A3";
TC.temperature_ini = 1273; % in Kelvin

%% Import alloy information and extract data
%load('bio_alloys/filtered_bio_df.mat');
load('data4\df_all.mat')
%Get compositions in molar fractions

T = df2;

for ii = 1:height(T)
    composition{ii} = get_composition(T.alloy{ii});
end
%Append compositions to table
T.composition = composition';

%% Find T0 temperature
for ii = 1:height(T)  
    fprintf("*** %d/%d: %s *** \n",ii,height(T),T.alloy{ii});
    tzero(ii) = batch_tzero(composition{ii},TC); 
end