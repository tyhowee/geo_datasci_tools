import os
from osgeo import gdal

def check_and_fix_geotiff(input_folder, output_folder):
    """
    Check and fix GeoTIFF files to ensure they meet the specified standards.
    Standards: GeoTIFF (8-bit RGBA), EPSG:4326, lossless compression (COMPRESS=LZW).
    Also, export a .txt file with the extent of each GeoTIFF in a JSON-compatible format.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.tif') or file_name.lower().endswith('.tiff'):
            input_path = os.path.join(input_folder, file_name)
            
            # Add "_processed" before the file extension for the output filename
            base_name, ext = os.path.splitext(file_name)
            output_file_name = f"{base_name}_processed{ext}"
            output_path = os.path.join(output_folder, output_file_name)
            extent_file_path = os.path.join(output_folder, f"{base_name}_extent.txt")

            # Open the input GeoTIFF file
            ds = gdal.Open(input_path)
            if ds is None:
                print(f"Could not open {file_name}, skipping...")
                continue

            # Check CRS and reproject to EPSG:4326 if necessary
            crs = ds.GetProjection()
            target_crs = 'EPSG:4326'
            reprojected_ds_path = None
            converted_ds_path = None
            if target_crs not in crs:
                print(f"{file_name}: Reprojecting to EPSG:4326...")
                warp_options = gdal.WarpOptions(dstSRS=target_crs)
                reprojected_ds_path = f"/vsimem/{base_name}_reprojected.tif"
                reprojected_ds = gdal.Warp(reprojected_ds_path, ds, options=warp_options)
                if reprojected_ds is None:
                    print(f"{file_name}: Failed to reproject to EPSG:4326, skipping...")
                    continue
                ds = reprojected_ds

            # Check band count and data type for 8-bit RGBA
            band_count = ds.RasterCount
            data_type = ds.GetRasterBand(1).DataType
            if band_count != 4 or data_type != gdal.GDT_Byte:
                print(f"{file_name}: Converting to 8-bit RGBA...")
                translate_options = gdal.TranslateOptions(format='GTiff', outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=LZW"])
                converted_ds_path = f"/vsimem/{base_name}_converted.tif"
                converted_ds = gdal.Translate(converted_ds_path, ds, options=translate_options)
                if converted_ds is None:
                    print(f"{file_name}: Failed to convert to 8-bit RGBA, skipping...")
                    continue
                ds = converted_ds

            # Export the corrected GeoTIFF with LZW compression
            print(f"{file_name}: Saving as {output_file_name} with LZW compression...")
            gdal.Translate(output_path, ds, creationOptions=["COMPRESS=LZW"])

            # Extract and write extent to a .txt file
            geo_transform = ds.GetGeoTransform()
            min_x = geo_transform[0]
            max_y = geo_transform[3]
            max_x = min_x + geo_transform[1] * ds.RasterXSize
            min_y = max_y + geo_transform[5] * ds.RasterYSize
            extent_json = [[min_x, min_y], [max_x, max_y]]
            with open(extent_file_path, 'w') as extent_file:
                extent_file.write(str(extent_json))
            print(f"{file_name}: Extent written to {extent_file_path}")

            # Clean up in-memory datasets
            if reprojected_ds_path:
                gdal.Unlink(reprojected_ds_path)
            if converted_ds_path:
                gdal.Unlink(converted_ds_path)

            ds = None  # Close the dataset to release resources

if __name__ == "__main__":
    input_folder = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\ICB_data\geospatial_data\mapbox_tiffs"
    output_folder = input_folder
    check_and_fix_geotiff(input_folder, output_folder)
