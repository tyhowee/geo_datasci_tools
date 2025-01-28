import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from scipy.ndimage import median_filter

# Set default percentile threshold
percentile_threshold = 95  # User can edit this value as needed

# Prompt the user to select the input NetCDF file
root = Tk()
root.withdraw()  # Hide the main window
root.attributes("-topmost", True)  # Bring the dialog to the front
netcdf_path = askopenfilename(
    title="Select NetCDF file",
    filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")]
)
root.update()  # Update to make sure the selection goes through

if not netcdf_path:
    print("No file selected. Exiting.")
    exit()

# Load the NetCDF file and extract the combined_layers variable
ds = xr.open_dataset(netcdf_path)
if "combined_layers" not in ds:
    print("Error: 'combined_layers' variable not found in the NetCDF file.")
    exit()

# Extract and stack the layers
combined_layers = ds["combined_layers"].values  # Shape: (layers, y, x)
average_array = np.mean(combined_layers, axis=0)
min_array = np.min(combined_layers, axis=0)
max_array = np.max(combined_layers, axis=0)
std_dev_array = np.std(combined_layers, axis=0)

# Calculate CV map (handling division by zero where mean is zero)
cv_array = np.divide(std_dev_array, average_array, out=np.zeros_like(std_dev_array), where=average_array != 0)

# Calculate the threshold value based on the set percentile threshold
threshold_value = np.percentile(average_array, percentile_threshold)

# Create a thresholded array with values below the threshold set to zero
thresholded_array = np.where(average_array >= threshold_value, average_array, 0)

# Apply median filter to the thresholded array
window_size = 20  # Adjust window size as needed for smoothing
smoothed_array = median_filter(thresholded_array, size=window_size)

# Prompt the user to select the output NetCDF file path and name
output_file = asksaveasfilename(
    title="Save Averaged Layers",
    defaultextension=".nc",
    filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")]
)

if not output_file:
    print("No output file selected. Exiting.")
    exit()

#Reverse y-coords
print("Reversing y-axis to correct orientation.")
average_array = np.flipud(average_array)
min_array = np.flipud(min_array)
max_array = np.flipud(max_array)
std_dev_array = np.flipud(std_dev_array)
cv_array = np.flipud(cv_array)
thresholded_array = np.flipud(thresholded_array)
smoothed_array = np.flipud(smoothed_array)
y_coords = ds.coords["y"].values[::-1]  # Reverse y-coordinates

# Save results back to a new NetCDF file
output_ds = xr.Dataset(
    {
        "average_array": (["y", "x"], average_array),
        "min_array": (["y", "x"], min_array),
        "max_array": (["y", "x"], max_array),
        "std_dev_array": (["y", "x"], std_dev_array),
        "cv_array": (["y", "x"], cv_array),
        "thresholded_array": (["y", "x"], thresholded_array),
        "smoothed_array": (["y", "x"], smoothed_array)
    },
    coords={"y": y_coords, "x": ds.coords["x"]}
)
output_ds.to_netcdf(output_file)
print(f"Results saved to '{output_file}'")

# Plot the smoothed, thresholded probability map
plt.figure(figsize=(14, 10))
plt.imshow(smoothed_array, cmap='gray')
plt.colorbar(label='Value')
plt.title('Smoothed Thresholded Average Map', fontsize=15)
plt.axis('off')
plt.tight_layout()

# Save the plot as a PNG
png_file_path = output_file.replace(".nc", ".png")
plt.savefig(png_file_path, format='png')
print(f"Averaged map image saved as '{png_file_path}'")

plt.show()