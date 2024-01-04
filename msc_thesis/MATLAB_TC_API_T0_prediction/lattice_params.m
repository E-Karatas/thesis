clear variables; home; close all

load('data4/df_Fe_nr.mat');

% Initialize the table with zeros
temp = array2table(zeros(149, 7));

% Assign variable names as a cell array
temp.Properties.VariableNames = {'a_BCC', 'a_HCP', 'b_HCP', 'c_HCP', 'a_ortho', 'b_ortho', 'c_ortho'};

% Lattice parameter for beta (BCC) in Å
temp.a_BCC = 3.274 + 0.043*df2.Nb - 0.1259*df2.Mo - 0.2*df2.V - 0.55*df2.Fe - 0.15*df2.W + 0.044*df2.Ta + 0.11*df2.Sn - 0.4*df2.Cr + 0.31*df2.Zr;

% Lattice parameters for alpha' (HCP) in Å
temp.a_HCP = 2.959 - 0.083*df2.Fe - 1.3*df2.W + 0.22*df2.Zr - 0.26*df2.Cr - df2.V;
temp.b_HCP = sqrt(3) * temp.a_HCP;
temp.c_HCP = 4.68 + 0.2381*df2.Mo + 0.5*df2.Nb + 0.5*df2.Ta - 0.4*df2.W - 0.4*df2.V + 0.1*df2.Fe + 0.5*df2.Zr - 1.6*df2.Cr;

% Lattice parameters for alpha'' (orthorhombic) in Å
temp.a_ortho = 2.89 + 1.37*df2.Nb + 3*df2.Mo + 0.6*df2.Ta + 2.43*df2.W + 2.05*df2.Sn + df2.V + 3*df2.Cr + 0.9*df2.Zr + 0.083*df2.Al + 0.0223*df2.Fe;
temp.b_ortho = 3.02*sqrt(3) - 1.68*sqrt(3)*df2.Nb - 3*sqrt(3)*df2.Mo - 1.21*df2.Ta - 1.42*sqrt(3)*df2.W - 2.64*df2.Sn - 0.28*df2.Zr - sqrt(3)*df2.V - 5.67*df2.Cr - 0.018*df2.Al + 0.0504*df2.Fe;
temp.c_ortho = 4.734 - 0.184 * df2.Nb - 1.5 * df2.Mo - 0.4 * df2.Ta - 0.57 * df2.W + 0.23 * df2.Sn + 0.4 * df2.V - 2.67 * df2.Cr + 0.3 * df2.Zr - 0.0535 * df2.Al - 0.0326 * df2.Fe;

% Append columns from temp to df2 after the 'Fe_eqnr' column
df2 = addvars(df2, temp.a_BCC, temp.a_HCP, temp.b_HCP, temp.c_HCP, temp.a_ortho, temp.b_ortho, temp.c_ortho, 'After', 'Fe_eqnr', 'NewVariableNames', {'a_BCC', 'a_HCP', 'b_HCP', 'c_HCP', 'a_ortho', 'b_ortho', 'c_ortho'});

% Initialize the table with zeros
temp2 = array2table(zeros(149, 16));

% Assign variable names as a cell array
temp2.Properties.VariableNames = {'lambda1_HCP', 'lambda2_HCP', 'lambda3_HCP', 'lambda1_ortho', 'lambda2_ortho', 'lambda3_ortho', 'e1_HCP', 'e2_HCP', 'e3_HCP', 'e_HCP', 'e1_ortho', 'e2_ortho', 'e3_ortho', 'e_ortho', 'dV_HCP', 'dV_ortho'};

% Calculate the eigenvalues of the strain tensor describing the BCC to HCP transformation
temp2.lambda1_HCP = (temp.a_HCP - temp.a_BCC) ./ temp.a_BCC;
temp2.lambda2_HCP = (temp.b_HCP - sqrt(2) * temp.a_BCC) ./ (sqrt(2) * temp.a_BCC);
temp2.lambda3_HCP = (temp.c_HCP - sqrt(2) * temp.a_BCC) ./ (sqrt(2) * temp.a_BCC);

% Calculate the eigenvalues of the strain tensor describing the BCC to Orthorhombic transformation
temp2.lambda1_ortho = (temp.a_ortho - temp.a_BCC) ./ temp.a_BCC;
temp2.lambda2_ortho = (temp.b_ortho - sqrt(2) * temp.a_BCC) ./ (sqrt(2) * temp.a_BCC);
temp2.lambda3_ortho = (temp.c_ortho - sqrt(2) * temp.a_BCC) ./ (sqrt(2) * temp.a_BCC);

% Calculate the eigenvalues of the deviatoric component of the strain
% tensor for the BCC to HCP transformation
temp2.e1_HCP = temp2.lambda1_HCP - 1/3 * (temp2.lambda1_HCP + temp2.lambda2_HCP + temp2.lambda3_HCP);
temp2.e2_HCP = temp2.lambda2_HCP - 1/3 * (temp2.lambda1_HCP + temp2.lambda2_HCP + temp2.lambda3_HCP);
temp2.e3_HCP = temp2.lambda3_HCP - 1/3 * (temp2.lambda1_HCP + temp2.lambda2_HCP + temp2.lambda3_HCP);

% Calculate the transformation strain for the BCC to HCP transformation
temp2.e_HCP = sqrt(1/6 * ( (temp2.e1_HCP - temp2.e2_HCP).^2 + (temp2.e2_HCP - temp2.e3_HCP).^2 + (temp2.e3_HCP - temp2.e1_HCP).^2 ) );

% Calculate the eigenvalues of the deviatoric component of the strain
% tensor for the BCC to orthorhombic transformation
temp2.e1_ortho = temp2.lambda1_ortho - 1/3 * (temp2.lambda1_ortho + temp2.lambda2_ortho + temp2.lambda3_ortho);
temp2.e2_ortho = temp2.lambda2_ortho - 1/3 * (temp2.lambda1_ortho + temp2.lambda2_ortho + temp2.lambda3_ortho);
temp2.e3_ortho = temp2.lambda3_ortho - 1/3 * (temp2.lambda1_ortho + temp2.lambda2_ortho + temp2.lambda3_ortho);

% Calculate the transformation strain for the BCC to orthorhombic
% transformation 
temp2.e_ortho = sqrt(1/6 * ( (temp2.e1_ortho - temp2.e2_ortho).^2 + (temp2.e2_ortho - temp2.e3_ortho).^2 + (temp2.e3_ortho - temp2.e1_ortho).^2 ) );

df2 = addvars(df2, temp2.lambda1_HCP, temp2.lambda2_HCP, temp2.lambda3_HCP, temp2.e1_HCP, temp2.e2_HCP, temp2.e3_HCP, temp2.e_HCP, ...
               temp2.lambda1_ortho, temp2.lambda2_ortho, temp2.lambda3_ortho, temp2.e1_ortho, temp2.e2_ortho, temp2.e3_ortho, temp2.e_ortho, ...
               'After', 'c_ortho', 'NewVariableNames', {'lambda1_HCP', 'lambda2_HCP', 'lambda3_HCP', 'e1_HCP', 'e2_HCP', 'e3_HCP', 'e_HCP', ...
                                                        'lambda1_ortho', 'lambda2_ortho', 'lambda3_ortho', 'e1_ortho', 'e2_ortho', 'e3_ortho', 'e_ortho'});

% Calculate ΔV/V for the BCC to HCP transformation
temp2.dV_HCP = ( (1/3) * (temp2.lambda1_HCP + temp2.lambda2_HCP + temp2.lambda3_HCP) + 1 ).^3 - 1;

% Calculate ΔV/V for the BCC to orthorhombic transformation
temp2.dV_ortho = ( (1/3) * (temp2.lambda1_ortho + temp2.lambda2_ortho + temp2.lambda3_ortho) + 1 ).^3 - 1;

% Append columns from temp2 to df2 after the 'e_ortho' column
df2 = addvars(df2, temp2.dV_HCP, temp2.dV_ortho, 'After', 'e_ortho', 'NewVariableNames', {'dV_HCP', 'dV_ortho'});

