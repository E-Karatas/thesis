clear variables; home; close all

load('data4/validation_set4.mat');
load('data/elements.mat');

df2 = bio_alloys_df;

% Initialize the table with zeros
temp = array2table(zeros(height(df2), 14));

% Assign variable names as a cell array
temp.Properties.VariableNames = {'alloy', 'Ti','Fe', 'Cr', 'Mo', 'V', 'W', 'Sn', 'Nb', 'Ta', 'Zr', 'Al', 'total', 'Fe_eqnr'};

temp.alloy = df2.alloy;

temp.Ti = df2.Ti*elements.atomic_mass('Ti');
temp.Fe = df2.Fe*elements.atomic_mass('Fe');
temp.Cr = df2.Cr*elements.atomic_mass('Cr');
temp.Mo = df2.Mo*elements.atomic_mass('Mo');
temp.V = df2.V*elements.atomic_mass('V');
temp.W = df2.W*elements.atomic_mass('W');
temp.Sn = df2.Sn*elements.atomic_mass('Sn');
temp.Nb = df2.Nb*elements.atomic_mass('Nb');
temp.Ta = df2.Ta*elements.atomic_mass('Ta');
temp.Zr = df2.Zr*elements.atomic_mass('Zr');
temp.Al = df2.Al*elements.atomic_mass('Al');

temp.total = temp.Ti + temp.Fe + temp.Cr + temp.Mo + temp.V + temp.W + temp.Sn + temp.Nb + temp.Ta + temp.Zr + temp.Al;

temp.Ti = 100*df2.Ti*elements.atomic_mass('Ti')./temp.total;
temp.Fe = 100*df2.Fe*elements.atomic_mass('Fe')./temp.total;
temp.Cr = 100*df2.Cr*elements.atomic_mass('Cr')./temp.total;
temp.Mo = 100*df2.Mo*elements.atomic_mass('Mo')./temp.total;
temp.V = 100*df2.V*elements.atomic_mass('V')./temp.total;
temp.W = 100*df2.W*elements.atomic_mass('W')./temp.total;
temp.Sn = 100*df2.Sn*elements.atomic_mass('Sn')./temp.total;
temp.Nb = 100*df2.Nb*elements.atomic_mass('Nb')./temp.total;
temp.Ta = 100*df2.Ta*elements.atomic_mass('Ta')./temp.total;
temp.Zr = 100*df2.Zr*elements.atomic_mass('Zr')./temp.total;
temp.Al = 100*df2.Al*elements.atomic_mass('Al')./temp.total;

temp.Fe_eqnr = 3.5*(temp.Fe/3.5 +temp.Cr/9 +temp.Mo/14 +temp.V/20 +temp.W/25 +temp.Sn/27 +temp.Nb/43 +temp.Ta/75 +temp.Zr/90 -temp.Al/18);

% Append the Fe_eqnr column from temp to df2
df2 = addvars(df2, temp.Fe_eqnr, 'After', 'T0', 'NewVariableName', 'Fe_eqnr');