function tzero = batch_tzero(x,TC)
%% Get compositions
alloying_elements = string(x.Properties.RowNames(x.content>0));
phases = [TC.parent_phase,TC.child_phase];
composition = x.content(x.content>0);
[~,max_ind] = max(composition);
dependent_element = alloying_elements(max_ind);
composition(max_ind) = [];
alloying_elements(max_ind) = [];
%% Start a TC session and set cache folder
session = tc_toolbox.TCToolbox();
[~,name,~] = fileparts(mfilename("fullpath"));
session.set_cache_folder(name + "_cache");
%% Define a system (specify database + elements + phases)
%Database and elements
if ~isempty(alloying_elements)
    system_builder = session.select_database_and_elements(TC.database, [dependent_element,alloying_elements']);
else
    system_builder = session.select_database_and_elements(TC.database, dependent_element);
end
%Phases
system_builder.without_default_phases(); %Reject all phases
for ii = 1:length(phases) %Add phases 1 by 1
    system_builder.select_phase(phases(ii));
end
%Get database information on system
sys = system_builder.get_system();
%% Set initial temperature and composition
batch_calculation = sys.with_batch_equilibrium_calculation();
batch_calculation.set_condition("T",TC.temperature_ini);
for ii = 1:length(alloying_elements)
    batch_calculation.set_condition(...
            tc_toolbox.ThermodynamicQuantity.mole_fraction_of_a_component(alloying_elements(ii)), composition(ii));
end

for ph = 1:length(phases)    
    batch_calculation.set_phase_to_suspended("*");
    batch_calculation.set_phase_to_entered(phases(ph));
    
    batch_calculation.disable_global_minimization();
    
    %% Define batch calculation job
    list_of_T = linspace(1500, 100, 1400);
    equilibria = cell(1,length(list_of_T));
    
    i = 1;
    for T = list_of_T
        equilibria{i} = {{"T", T}};
        i = i + 1;
    end
    batch_calculation.set_conditions_for_equilibria(equilibria);
    
    %% Calculate and retrieve results
    results = batch_calculation.calculate(strcat("G(",phases(ph), ")"), 1000);
    G(:,ph) = results.get_values_of(strcat("G(",phases(ph), ")"));
end

dg_chem = G(:,1) - G(:,2);

%% calculate t0
[~,min_ind] = min(abs(dg_chem));
tzero = list_of_T(min_ind);
fprintf("T0: %.2fK, dg_chem: %.2f",tzero,dg_chem(min_ind));


