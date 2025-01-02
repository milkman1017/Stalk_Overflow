import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV files
rf_yield = pd.read_csv('final_predictions_random_forest.csv')
nn_yield = pd.read_csv('final_yield_predictions.csv')
real_data = pd.read_csv('/home/wimahler/Stalk_Overflow/Training_data/1_Training_Trait_Data_2014_2023.csv')
real_data = real_data.dropna()

unique_years = real_data['Year'].unique()

# Create the plot
fig, ax = plt.subplots(figsize=(10, 7))  # Single plot

# Plot the Random Forest and Neural Net histograms

# Plot separate histograms for each year with different shades of orange
colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(unique_years)))  # Generate shades of orange
for i, year in enumerate(sorted(unique_years)):
    year_data = real_data[real_data['Year'] == year]
    ax.hist(year_data['Yield_Mg_ha'], bins=30, color=colors[i], alpha=0.3, density=True, label=f"{year}")

# ax.hist(rf_yield['Yield_Mg_ha'], bins=30, color='blue', alpha=0.7, density=True, label="Random Forest")
ax.hist(nn_yield['Yield_Mg_ha'], bins=30, color='green', alpha=0.7, density=True, label="Neural Net")
# Add titles and labels
ax.set_title('Yield Distributions')
ax.set_xlabel('Yield (Mg/ha)')
ax.set_ylabel('Density')

# Add legend
ax.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('distribution.png')
