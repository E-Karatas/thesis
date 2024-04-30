import pandas as pd
import os

# The problematic alloy is already dropped in this dataset
file_name = 'validation_set_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the saved DataFrame
df = pd.read_pickle(file_path)
df['1-Ti'] = 1 - df['Ti'] # alloying content

# Selecting all the features and 'T0' columns
features = ['alloy', 'T0', 'Fe_eqnr','e_ortho', 'dV_ortho', 'alloyType']
df3 = df[features]

# Convert DataFrame to LaTeX table format with longtable option and other formatting adjustments
latex_table = df3.to_latex(index=True, 
                           longtable=True,
                           header=features,  # Provide headers
                           bold_rows=True,  # Make table rows bold
                           caption='Caption for your table',  # Add a caption
                           label='tab:my_table'  # Add a label for referencing
                           )

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save the LaTeX table to a text file in the same directory as the script
output_file_path = os.path.join(script_dir, 'latex_table.txt')
with open(output_file_path, 'w') as file:
    file.write(latex_table)

print(f"LaTeX table saved to: {output_file_path}")
