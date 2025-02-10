import os
import numpy as np
from osgeo import gdal


def check_and_fix_geotiff(input_path, output_folder):
    """
    Check and fix GeoTIFF files to ensure they meet the specified standards.
    Standards: GeoTIFF (8-bit RGBA), EPSG:4326, lossless compression (COMPRESS=LZW).
    Also, export a .txt file with the extent of each GeoTIFF in a JSON-compatible format.
    Accepts either a file or a folder as input.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Determine if input is a file or a folder
    if os.path.isfile(input_path) and input_path.lower().endswith((".tif", ".tiff")):
        files_to_process = [input_path]
    elif os.path.isdir(input_path):
        files_to_process = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith((".tif", ".tiff"))
        ]
    else:
        print(f"Invalid input: {input_path} is neither a valid file nor a folder.")
        return

    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)
        output_path = os.path.join(output_folder, f"{base_name}_processed{ext}")
        extent_file_path = os.path.join(output_folder, f"{base_name}_extent.txt")

        ds = gdal.Open(file_path)
        if ds is None:
            print(f"Could not open {file_name}, skipping...")
            continue

        # Get NoData value
        band = ds.GetRasterBand(1)
        nodata_value = band.GetNoDataValue()

        # Check if CRS is EPSG:4326
        crs = ds.GetProjection()
        current_ds = ds
        if "EPSG:4326" not in crs:
            print(f"{file_name}: Reprojecting to EPSG:4326...")
            reprojected_ds_path = f"/vsimem/{base_name}_reprojected.tif"
            current_ds = gdal.Warp(
                reprojected_ds_path,
                ds,
                options=gdal.WarpOptions(
                    dstSRS="EPSG:4326",
                    format="GTiff",
                    creationOptions=["COMPRESS=LZW"],
                ),
            )
            if current_ds is None:
                print(f"{file_name}: Failed to reproject, skipping...")
                continue

        num_bands = current_ds.RasterCount

        if num_bands == 1:
            print(f"{file_name}: Converting singleband to normalized RGBA...")
            rgba_ds_path = f"/vsimem/{base_name}_rgba.tif"
            driver = gdal.GetDriverByName("GTiff")
            rgba_ds = driver.Create(
                rgba_ds_path,
                current_ds.RasterXSize,
                current_ds.RasterYSize,
                4,
                gdal.GDT_Byte,
                options=["COMPRESS=LZW"],
            )
            rgba_ds.SetGeoTransform(current_ds.GetGeoTransform())
            rgba_ds.SetProjection(current_ds.GetProjection())

            # Read singleband data
            band_data = current_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

            # Determine NoData mask
            if nodata_value is not None:
                nodata_mask = band_data == nodata_value
            else:
                nodata_mask = np.isnan(band_data) | (band_data < 0)

            # Normalize band values to range 0-255
            valid_mask = ~nodata_mask
            if np.any(valid_mask):
                min_val, max_val = np.min(band_data[valid_mask]), np.max(
                    band_data[valid_mask]
                )
                if min_val != max_val:
                    band_data[valid_mask] = (
                        (band_data[valid_mask] - min_val) / (max_val - min_val)
                    ) * 255

            # Convert NaNs and NoData values to 0
            band_data[nodata_mask] = 0
            band_data = np.clip(band_data, 0, 255).astype(np.uint8)

            # Generate alpha band (0 for NoData, 255 for valid pixels)
            alpha_band = np.where(nodata_mask, 0, 255).astype(np.uint8)

            # Write RGBA bands
            rgba_ds.GetRasterBand(1).WriteArray(band_data)  # Red
            rgba_ds.GetRasterBand(2).WriteArray(band_data)  # Green
            rgba_ds.GetRasterBand(3).WriteArray(band_data)  # Blue
            rgba_ds.GetRasterBand(4).WriteArray(alpha_band)  # Transparency

            current_ds = rgba_ds

        elif num_bands == 3:
            print(f"{file_name}: Adding alpha channel to RGB...")
            rgba_ds_path = f"/vsimem/{base_name}_rgba.tif"
            driver = gdal.GetDriverByName("GTiff")
            rgba_ds = driver.Create(
                rgba_ds_path,
                current_ds.RasterXSize,
                current_ds.RasterYSize,
                4,
                gdal.GDT_Byte,
                options=["COMPRESS=LZW"],
            )
            rgba_ds.SetGeoTransform(current_ds.GetGeoTransform())
            rgba_ds.SetProjection(current_ds.GetProjection())

            # Copy RGB bands
            for i in range(3):
                band_data = current_ds.GetRasterBand(i + 1).ReadAsArray()
                rgba_ds.GetRasterBand(i + 1).WriteArray(band_data)

            # Use NoData mask for alpha
            alpha_band = np.where(nodata_mask, 0, 255).astype(np.uint8)
            rgba_ds.GetRasterBand(4).WriteArray(alpha_band)  # Transparency

            current_ds = rgba_ds

        elif num_bands >= 4:
            print(
                f"{file_name}: Image already has 4 bands, ensuring LZW compression..."
            )
            rgba_ds_path = f"/vsimem/{base_name}_rgba.tif"
            driver = gdal.GetDriverByName("GTiff")
            rgba_ds = driver.CreateCopy(
                rgba_ds_path, current_ds, options=["COMPRESS=LZW"]
            )

            current_ds = rgba_ds

        # Save with LZW compression
        print(f"{file_name}: Saving with LZW compression...")
        gdal.Translate(
            output_path,
            current_ds,
            options=gdal.TranslateOptions(creationOptions=["COMPRESS=LZW"]),
        )

        # Write extent
        geo_transform = current_ds.GetGeoTransform()
        min_x = geo_transform[0]
        max_y = geo_transform[3]
        max_x = min_x + geo_transform[1] * current_ds.RasterXSize
        min_y = max_y + geo_transform[5] * current_ds.RasterYSize

        with open(extent_file_path, "w") as extent_file:
            extent_file.write(str([[min_x, min_y], [max_x, max_y]]))

        # Cleanup
        if current_ds != ds:
            gdal.Unlink(rgba_ds_path)
        ds = None
        current_ds = None


if __name__ == "__main__":
    input_path = "/Users/thowe/MinersAI Dropbox/Tyler Howe/AK_sample_project/PROCESSED/processed_data/TIFFs/alaska_geology.tif"  # Can be a file or folder
    output_folder = (
        input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
    )

    check_and_fix_geotiff(input_path, output_folder)
    print("Done processing GeoTIFFs.")
