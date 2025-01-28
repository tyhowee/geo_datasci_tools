import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the NetCDF file
#file_path = '/Users/thowe/MinersAI Dropbox/Tyler Howe/ICB_data/testing/uncorrelated_layers_1.nc' # Replace with your NetCDF file path
file_path = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\ICB_data\testing\uncorrelated_layers_3.nc"
ds = xr.open_dataset(file_path)

# Extract the combined_layers array and get layer names
combined_layers = ds["combined_layers"]
layer_names = combined_layers.layer.values  # Assumes layer names are stored in the 'layer' dimension

# Reshape data for each layer into a DataFrame with actual layer names
data = {layer_name: combined_layers.sel(layer=layer_name).values.flatten() for layer_name in layer_names}
df = pd.DataFrame(data)

samples = 500

# Downsample 
df_sampled = df.sample(n=samples, random_state=42)

# Calculate the correlation matrix
correlation_matrix = df_sampled.corr()

# Plot the diagonal correlation matrix
sns.set(style="white")
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(8, 6))  # Adjust figure size as needed
sns.heatmap(correlation_matrix, 
            mask=mask, 
            cmap="coolwarm", 
            annot=True, 
            square=True, 
            linewidths=0.5, 
            cbar_kws={"shrink": 0.8, "aspect": 30, "pad": 0.02},  # Adjust 'shrink' and add 'aspect' and 'pad'
            xticklabels=layer_names, 
            yticklabels=layer_names,
            vmin=-1, 
            vmax=1
            )

plt.title(f"Diagonal Correlation Matrix of Combined Layers ({samples} Random Samples)", pad=20)
plt.xticks(rotation=45, ha="right", fontsize=8)  # Rotate and align x-axis labels
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()  # Adjust layout to fit everything nicely
plt.show()
