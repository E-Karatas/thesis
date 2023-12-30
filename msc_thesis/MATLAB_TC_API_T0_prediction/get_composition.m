function composition = get_composition(compString,delimiter)
% Extract composition in weight percent from composition string
% and return it in molar fraction
%
% Input variables: 
% compString (example: "Ti-12Nb-3Zr")
% delimiter: delimiter string for different alloying contributions (default is "-")

%Set delimiter
if nargin == 1
    delimiter = "-";
end

%Initialize composition
load('data/elements.mat')
elements.Properties.VariableNames = {'atomic_number','atomic_mass'};
elements.content = zeros(height(elements),1);
composition = elements;
clear elements

%Get alloying contributions    
C = strsplit(compString,delimiter);
for ii = 2:length(C)
    element = extract(C{ii},lettersPattern);
    content = str2double(regexp(C{ii},'\d+\.?\d*','match'));
    composition.content(element{1}) = content;
end
composition.content(C{1}) = 100 - sum(composition.content);

%Convert to molar fraction
nr_atoms = composition.content./composition.atomic_mass;
composition.content = nr_atoms / sum(nr_atoms);

