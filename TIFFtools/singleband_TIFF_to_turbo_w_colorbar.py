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

        colormap = colormaps["turbo"]

        for file in files:
            if file.endswith(".tif"):
                input_path = os.path.join(root, file)
                output_raster_path = os.path.join(
                    output_folder, f"{os.path.splitext(file)[0]}_colored.tif"
                )
                colorbar_path = os.path.join(
                    output_folder, f"{os.path.splitext(file)[0]}_colorbar.png"
                )

                # Open the GeoTIFF
                dataset = gdal.Open(input_path)
                band = dataset.GetRasterBand(1)
                array = band.ReadAsArray()

                # Check if the input file has an alpha channel
                alpha_channel = None
                if (
                    dataset.RasterCount >= 4
                ):  # If there are at least 4 bands, assume the 4th is alpha
                    alpha_band = dataset.GetRasterBand(4)
                    alpha_channel = alpha_band.ReadAsArray().astype(np.uint8)

                # Normalize data
                if "cv" in file.lower():
                    max_value = np.max(array)
                    data_to_process = array / max_value if max_value > 0 else array
                else:
                    data_to_process = (array - np.min(array)) / (
                        np.max(array) - np.min(array)
                    )

                # Apply colormap
                rgba_array = (colormap(data_to_process) * 255).astype(np.uint8)

                # Retain original alpha if present, otherwise apply logic
                if alpha_channel is not None:
                    rgba_array[:, :, 3] = alpha_channel  # Use the existing alpha
                else:
                    rgba_array[:, :, 3] = np.where(array == 0, 0, 255).astype(
                        np.uint8
                    )  # Apply logic for transparency

                # Export RGBA GeoTIFF
                driver = gdal.GetDriverByName("GTiff")
                out_raster = driver.Create(
                    output_raster_path,
                    dataset.RasterXSize,
                    dataset.RasterYSize,
                    4,
                    gdal.GDT_Byte,
                )
                for i in range(4):  # RGBA channels
                    out_band = out_raster.GetRasterBand(i + 1)
                    out_band.WriteArray(rgba_array[:, :, i])
                out_raster.SetGeoTransform(dataset.GetGeoTransform())
                out_raster.SetProjection(dataset.GetProjection())
                out_raster.FlushCache()

                # Create colorbar
                norm = Normalize(
                    vmin=0, vmax=1 if "cv" not in file.lower() else max_value
                )
                sm = ScalarMappable(norm=norm, cmap=colormap)

                fig, ax = plt.subplots(figsize=(8, 1))
                fig.subplots_adjust(bottom=0.5)

                colorbar_label = (
                    # "Coefficient of Variation"
                    # if "cv" in file.lower()
                    # else "Relative Deposit Probability"
                    f"{os.path.splitext(file)[0]} Values"
                )

                cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")
                cbar.set_label(colorbar_label)
                plt.savefig(colorbar_path, bbox_inches="tight")
                plt.close()

                print(f"Processed: {file} -> {output_folder}")

    print("Processing complete.")


# Example usage
input_dir = "/Users/thowe/MinersAI Dropbox/Tyler Howe/AK_sample_project/PROCESSED/processed_data"
apply_turbo_colormap(input_dir)
