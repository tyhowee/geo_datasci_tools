from osgeo import gdal, ogr
import os

# Enable GDAL exceptions
gdal.UseExceptions()


def convert_geotiff_to_mbtiles(input_geotiff, output_mbtiles):
    """
    Converts a GeoTIFF file to an MBTiles file compatible with Mapboard GIS by:
    1. Expanding color channels if needed.
    2. Reprojecting to Web Mercator.
    3. Converting to MBTiles.
    4. Adding overviews for multi-resolution support.

    Parameters:
    input_geotiff (str): Path to the input GeoTIFF file.
    output_mbtiles (str): Path to the output MBTiles file.
    """
    try:
        # Step 1: Expand color channels (RGB if needed)
        expanded_geotiff = "expanded.tif"
        gdal.Translate(
            expanded_geotiff,
            input_geotiff,
            options=gdal.TranslateOptions(format="GTiff", creationOptions=["TILED=YES", "COMPRESS=LZW", "PHOTOMETRIC=RGB"])
        )

        # Step 2: Reproject to Web Mercator
        reprojected_geotiff = "reprojected.tif"
        gdal.Warp(
            reprojected_geotiff,
            expanded_geotiff,
            dstSRS="EPSG:3857",
            format="GTiff"
        )

        # Step 3: Convert to MBTiles
        gdal.Translate(
            output_mbtiles,
            reprojected_geotiff,
            format='MBTILES'
        )

        # Step 4: Add overviews for multi-resolution support
        gdal.Open(output_mbtiles, gdal.GA_Update).BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64, 128, 256])

        # Clean up intermediate files
        os.remove(expanded_geotiff)
        os.remove(reprojected_geotiff)

        print(f"Conversion complete. MBTiles saved at {output_mbtiles}")

    except RuntimeError as e:
        print(f"An error occurred: {e}")


# Paths
input_geotiff = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\ICB_data\geospatial_data\IdahoCobaltBeltGeoMap_GR.tif"
output_mbtiles = r"C:\Users\TyHow\MinersAI Dropbox\Tyler Howe\ICB_data\geospatial_data\ICB_GeoMap.mbtiles"

# Run the reprojection and conversion
convert_geotiff_to_mbtiles(input_geotiff, output_mbtiles)