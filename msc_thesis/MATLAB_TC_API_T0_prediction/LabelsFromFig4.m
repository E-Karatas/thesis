% adding the labels from figure 4 of the article
% Modelling martensitic transformation in titanium alloys: 
% The influence of temperature and deformation 
% to our dataset
clear global; home; close all

load('data4/df_all.mat');

% Create a cell array of empty strings
initialAlloyType = repmat({''}, height(df2), 1);
problematicAlloy = zeros(height(df2), 1);

% Insert 'AlloyType' after 'Deformation_mechanism_simple'
df2 = addvars(df2, initialAlloyType, 'After', 'Deformation_mechanism_simple', 'NewVariableNames', 'alloyType');
df2 = addvars(df2, problematicAlloy, 'After', 'alloyType', 'NewVariableNames', 'LookIntoBehaviour');

df2.alloyType(9) = {'Not known'};

% Martensitic at room T
df2.alloyType(33:37) = {'Martensitic at room T'};
df2.alloyType(47:51) = {'Martensitic at room T'};
df2.alloyType(64) = {'Martensitic at room T'};
df2.alloyType(83:92) = {'Martensitic at room T'};
df2.alloyType(95:113) = {'Martensitic at room T'};
df2.alloyType(115) = {'Martensitic at room T'};
df2.alloyType(119:122) = {'Martensitic at room T'};
df2.alloyType(127) = {'Martensitic at room T'};
df2.alloyType(138) = {'Martensitic at room T'};
df2.alloyType(145:146) = {'Martensitic at room T'};

% TRIP
df2.alloyType(5:6) = {'TRIP'};
df2.alloyType(10:12) = {'TRIP'};
df2.alloyType(15:16) = {'TRIP'};
df2.alloyType(24:26) = {'TRIP'};
df2.alloyType(31) = {'TRIP'};
df2.alloyType(38) = {'TRIP'};
df2.alloyType(41) = {'TRIP'};
df2.alloyType(44:46) = {'TRIP'};
df2.alloyType(52:54) = {'TRIP'};
df2.alloyType(59) = {'TRIP'};
df2.alloyType(60:61) = {'TRIP'};
df2.alloyType(77) = {'TRIP'};
df2.alloyType(80:82) = {'TRIP'};
df2.alloyType(114) = {'TRIP'};
df2.alloyType(116:117) = {'TRIP'};
df2.alloyType(123:124) = {'TRIP'};
df2.alloyType(128:129) = {'TRIP'};
df2.alloyType(130) = {'TRIP'};
df2.alloyType(132) = {'TRIP'};
df2.alloyType(134:136) = {'TRIP'};
df2.alloyType(144) = {'TRIP'};
df2.alloyType(148) = {'TRIP'};

% Superelastic
df2.alloyType(13:14) = {'Superelastic'};
df2.alloyType(17) = {'Superelastic'};
df2.alloyType(19:23) = {'Superelastic'};
df2.alloyType(26:30) = {'Superelastic'};
df2.alloyType(32) = {'Superelastic'};
df2.alloyType(62) = {'Superelastic'};
df2.alloyType(65:66) = {'Superelastic'};
df2.alloyType(125) = {'Superelastic'};
df2.alloyType(147) = {'Superelastic'};
df2.alloyType(137) = {'Superelastic'};
df2.alloyType(149) = {'Superelastic'};

% Slip or TWIP
df2.alloyType(1:4) = {'Slip or TWIP'};
df2.alloyType(7:8) = {'Slip or TWIP'};
df2.alloyType(18) = {'Slip or TWIP'};
df2.alloyType(39:40) = {'Slip or TWIP'};
df2.alloyType(42:43) = {'Slip or TWIP'};
df2.alloyType(55:58) = {'Slip or TWIP'};
df2.alloyType(63) = {'Slip or TWIP'};
df2.alloyType(67:76) = {'Slip or TWIP'};
df2.alloyType(78:79) = {'Slip or TWIP'};
df2.alloyType(93:94) = {'Slip or TWIP'};
df2.alloyType(118) = {'Slip or TWIP'};
df2.alloyType(126) = {'Slip or TWIP'};
df2.alloyType(131) = {'Slip or TWIP'};
df2.alloyType(133) = {'Slip or TWIP'};
df2.alloyType(139:143) = {'Slip or TWIP'};