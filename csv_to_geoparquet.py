import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Define input and output files
csv_file = '/Users/thowe/MinersAI Dropbox/Tyler Howe/ICB_data/geospatial_data/rad:mag/USA_Idaho_Cobalt_Belt_Radiometric.csv'
output_file = '/Users/thowe/MinersAI Dropbox/Tyler Howe/ICB_data/geospatial_data/rad:mag/ICB_rad_processed.parquet'

print(f'Input file selected: {csv_file}')

# Define columns to keep
columns_of_interest = ["LAT", "LONG", "TOT_COR", "K_COR", 'U_COR', 'Th_COR', "EFFH"]

print(f"\nSelected columns: {columns_of_interest}\nProcessing file......")

# Step 1: Load CSV and filter columns
data = pd.read_csv(csv_file, usecols=columns_of_interest, low_memory=False)

# Step 2: Convert numeric columns to appropriate data types
numeric_columns = ["TOT_COR", "K_COR", "U_COR", "Th_COR", "EFFH"]
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Step 3: Drop rows with invalid data in numeric columns
data = data.dropna(subset=numeric_columns)

# Step 4: Create geometry column
data["geometry"] = data.apply(lambda row: Point(row["LONG"], row["LAT"]), axis=1)

# Step 5: Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")  # WGS84

# Step 6: Save to GeoParquet or GeoJSON
gdf.to_parquet(output_file)  # For GeoParquet
# gdf.to_file(output_file, driver="GeoJSON")  # Uncomment for GeoJSON - filesize MUCH larger!!

print(f'File successfully exported at {output_file}')