import xarray as xr
import matplotlib.pyplot as plt

# Load the NetCDF file
file_path = '/Users/thowe/MinersAI Dropbox/Tyler Howe/ICB_data/cube_testing/all_RS_layers.nc'
data = xr.open_dataset(file_path)

# Select the 'combined_layers' variable
layers = data['combined_layers']

# Calculate the percentage of pixels with value 0 in each layer
zero_percentage = [(layer == 0).sum().item() / layer.size * 100 for layer in layers]

# Plot the results as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(range(len(zero_percentage)), zero_percentage)
plt.xlabel("Layer Index")
plt.ylabel("Percentage of Pixels with Value 0")
plt.title("Percentage of Pixels with Value 0 in Each Layer")
plt.xticks(range(len(zero_percentage)), range(len(zero_percentage)), rotation=90)  # Show each layer index
plt.tight_layout()
plt.show()