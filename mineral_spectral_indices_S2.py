import rasterio
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


# Function to compute a spectral index
def compute_index(numerator, denominator):
    """Calculate a normalized difference index while handling NaNs and Infs."""
    index = (numerator - denominator) / (numerator + denominator)
    index = np.nan_to_num(index, nan=0, posinf=1, neginf=0)  # Replace invalid values
    return index


# Set Your Parent Folder
parent_folder = "/Users/thowe/MinersAI Dropbox/Tyler Howe/AK_sample_project/UNPROCESSED/misc/spectral/R128"

# Define required Sentinel-2 bands (keys) and search for them in filenames
required_bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]  # B8A = NIR at 20m
band_files = {}

# Search for required bands in filenames
for band in required_bands:
    file_match = glob.glob(
        os.path.join(parent_folder, f"*{band}*20m.jp2")
    )  # Look for "Bxx" in filename
    if file_match:
        band_files[band] = file_match[0]  # Take the first match found
    else:
        print(f"⚠ Warning: {band} not found in {parent_folder}")

# Ensure all bands are available
if len(band_files) < len(required_bands):
    missing = [b for b in required_bands if b not in band_files]
    raise FileNotFoundError(f"🚨 Missing bands: {missing}. Check your file names.")

# Read bands into memory and extract geospatial metadata
bands = {}
with rasterio.open(
    band_files["B04"]
) as src:  # Use B04 (Red) as reference for geospatial data
    profile = src.profile  # Get metadata (CRS, transform, etc.)

for band, file in band_files.items():
    with rasterio.open(file) as src:
        bands[band] = src.read(1).astype(np.float32)  # Read band as float32

# Compute Spectral Indices
spectral_indices = {
    "Iron_Oxide_Index": compute_index(
        bands["B04"], bands["B02"]
    ),  # (Red - Blue) / (Red + Blue)
    "Clay_Minerals_Index": bands["B11"] / bands["B12"],  # SWIR1 / SWIR2
    "Silica_Index": (bands["B12"] - bands["B8A"])
    / (bands["B12"] + bands["B8A"]),  # (SWIR2 - NIR) / (SWIR2 + NIR)
    "Ferric_Iron_Alteration_Index": (bands["B12"] / bands["B8A"])
    + (bands["B03"] / bands["B04"]),  # (SWIR2 / NIR) + (Green / Red)
}

# Output Directory for GeoTIFFs
# output_dir = os.path.join(parent_folder, "spectral_indices")
output_dir = "/Users/thowe/MinersAI Dropbox/Tyler Howe/AK_sample_project/UNPROCESSED/misc_processing_files/spectral/R128"
os.makedirs(output_dir, exist_ok=True)

# Save each index as a georeferenced GeoTIFF
for index_name, index_array in spectral_indices.items():
    output_file = os.path.join(output_dir, f"{index_name}.tif")

    # Update profile for output raster (match georeferencing of B04)
    profile.update(
        dtype=rasterio.uint16,  # Use UInt16 for better storage
        count=1,
        compress="lzw",  # Apply compression
        driver="GTiff",
    )

    # Replace NaN, Inf and scale to uint16 (0–65535 range)
    index_array = np.nan_to_num(index_array, nan=0, posinf=1, neginf=0)
    index_array = np.clip(index_array * 10000, 0, 65535).astype(np.uint16)

    # Write to GeoTIFF with correct CRS and transform
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(index_array, 1)

    print(f"Saved {index_name} to {output_file}")

# Display one spectral index (e.g., Iron Oxide)
plt.imshow(spectral_indices["Iron_Oxide_Index"], cmap="RdYlBu")
plt.colorbar()
plt.title("Iron Oxide Index")
plt.show()
