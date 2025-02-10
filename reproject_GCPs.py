import pandas as pd
from pyproj import Transformer

# Define input/output CRS (change EPSG codes accordingly)
input_crs = "EPSG:4326"  # Original CRS
output_crs = "EPSG:9822"  # Target CRS
transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)

# Load .points file
input_file = "/Users/thowe/MinersAI Dropbox/Tyler Howe/AK_sample_project/UNPROCESSED/geology/Geologic Map of Southeastern Alaska.tif.points"
output_file = "/Users/thowe/MinersAI Dropbox/Tyler Howe/AK_sample_project/UNPROCESSED/geology/Geologic Map of Southeastern Alaska.tif_REPROJECTED.points"

# Read the .points file
with open(input_file, "r") as f:
    lines = f.readlines()

# Process and reproject
new_lines = []
for line in lines:
    line = line.strip()

    # Skip empty lines and comment lines
    if not line or line.startswith("#"):
        new_lines.append(line + "\n")  # Preserve header/comments
        continue

    parts = line.split()  # Assumes space-separated values
    if len(parts) < 5:
        continue  # Skip malformed lines

    try:
        # Convert to proper types
        mapX, mapY, pixelX, pixelY = map(float, parts[:4])
        enable = parts[4]  # Keep as string or convert to int if necessary

        # Reproject coordinates
        newX, newY = transformer.transform(mapX, mapY)

        # Recreate the line with reprojected coordinates
        new_lines.append(f"{newX} {newY} {pixelX} {pixelY} {enable}\n")

    except ValueError:
        print(f"Skipping invalid line: {line}")

# Save the new .points file
with open(output_file, "w") as f:
    f.writelines(new_lines)

print(f"Reprojected GCPs saved to {output_file}")
