import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import os

def export_layers_as_png(grayscale=False):
    # Initialize Tkinter and prompt for NetCDF file
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    netcdf_file = askopenfilename(
        title="Select the NetCDF file",
        filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")]
    )
    
    if not netcdf_file:
        print("No NetCDF file selected.")
        return
    
    # Prompt for output folder
    output_folder = askdirectory(
        title="Select output folder for PNG files"
    )
    
    if not output_folder:
        print("No output folder selected.")
        return
    
    # Load the NetCDF file
    data_xr = xr.open_dataset(netcdf_file)
    
    # Choose colormap
    colormap = 'gray' if grayscale else 'viridis'  # Use 'gray' for grayscale, 'viridis' for color
    
    # Iterate over each layer in the dataset
    for layer_name in data_xr['combined_layers'].coords['layer'].values:
        layer_data = data_xr['combined_layers'].sel(layer=layer_name).values
        
        plt.figure()
        plt.imshow(layer_data, cmap=colormap)
        plt.axis('off')  # Hide axes for a clean image
        output_path = os.path.join(output_folder, f"{layer_name}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved layer {layer_name} as PNG with {colormap} colormap.")

    print("All layers have been exported as PNG files.")
    root.destroy()

# Run the function, set grayscale to True for grayscale images, False for color
export_layers_as_png(grayscale=False)  # Set to False for color (same scale for all)
