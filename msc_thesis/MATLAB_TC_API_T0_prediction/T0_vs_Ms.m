clear variables; home; close all

load('data/alloy_data7.mat');

logicalIndex = T.Ms ~= 0;

df = T(logicalIndex, :);

% Scatter plot
scatter(df.T0, df.Ms);

% Set axis labels
xlabel('T0 [K]');
ylabel('Ms [K]');
title('Ms vs T0');

% Add a line y = x
hold on;  % This keeps the existing plot while adding the new line
line([min(df.T0), max(df.T0)], [min(df.T0), max(df.T0)], 'Color', 'red', 'LineStyle', '--');

% Legend
legend('Data', 'y = x', 'Location', 'Best');

hold off; % Release the hold on the plot