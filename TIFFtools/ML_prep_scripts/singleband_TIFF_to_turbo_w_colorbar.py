import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps

gdal.UseExceptions()

def apply_turbo_colormap(input_folder, subfolder_name="RGBA_outputs"):
    for root, _, files in os.walk(input_folder):
        # Create a subfolder for processed outputs in each folder
        output_folder = os.path.join(root, subfolder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Use the updated colormap function
        colormap = colormaps["turbo"]

        for file in files:
            if file.endswith(".tif"):
                input_path = os.path.join(root, file)
                output_raster_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_colored.tif")
                colorbar_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_colorbar.png")

                # Open the GeoTIFF
                dataset = gdal.Open(input_path)
                band = dataset.GetRasterBand(1)
                array = band.ReadAsArray()

                # Decide whether to normalize based on filename
                if "cv" in file.lower():
                    # Normalize to [0, max(array)]
                    max_value = np.max(array)
                    data_to_process = array / max_value if max_value > 0 else array
                else:
                    # Normalize the data to 0-1 range
                    data_to_process = (array - np.min(array)) / (np.max(array) - np.min(array))

                # Apply the colormap
                rgba_array = (colormap(data_to_process) * 255).astype(np.uint8)

                # Clamp values to 0–255 to ensure 8-bit validity
                rgba_array = np.clip(rgba_array, 0, 255).astype(np.uint8)

                # Add transparency for 0 values if "smoothed" is in the filename
                if "smoothed" in file.lower():
                    alpha_channel = np.where(array == 0, 0, 255).astype(np.uint8)
                    rgba_array[:, :, 3] = alpha_channel
                else:
                    rgba_array[:, :, 3] = 255  # Fully opaque if not "smoothed"

                # Export RGBA GeoTIFF
                driver = gdal.GetDriverByName('GTiff')
                out_raster = driver.Create(
                    output_raster_path,
                    dataset.RasterXSize,
                    dataset.RasterYSize,
                    4,
                    gdal.GDT_Byte
                )
                for i in range(4):  # RGBA channels
                    out_band = out_raster.GetRasterBand(i + 1)
                    out_band.WriteArray(rgba_array[:, :, i])
                out_raster.SetGeoTransform(dataset.GetGeoTransform())
                out_raster.SetProjection(dataset.GetProjection())
                out_raster.FlushCache()

                # Create colorbar
                norm = Normalize(vmin=0, vmax=1 if "cv" not in file.lower() else max_value)
                sm = ScalarMappable(norm=norm, cmap=colormap)

                fig, ax = plt.subplots(figsize=(8, 1))
                fig.subplots_adjust(bottom=0.5)

                # Determine colorbar label based on filename
                if "cv" in file.lower():
                    colorbar_label = "Coefficient of Variation"
                else:
                    colorbar_label = "Relative Deposit Probability"

                cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
                cbar.set_label(colorbar_label)
                plt.savefig(colorbar_path, bbox_inches='tight')
                plt.close()

                # Simplified print statement
                print(f"Processed: {file} -> {output_folder}")

    print("Processing complete.")

# Example usage
input_dir = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\ICB_data\50km_faults\avg\product_package"
apply_turbo_colormap(input_dir)
