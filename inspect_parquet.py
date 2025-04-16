import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Replace 'your_file.parquet' with the path to your Parquet file
file_path = '/Users/thowe/MinersAI Dropbox/Tyler Howe/QuebecProject/geo_data/output.parquet'

print(f'\nInspecting file: {file_path}\n')

# Load the Parquet file
df = pd.read_parquet(file_path)  # Can change to gpd.read_parquet for GeoParquet files

# Display general information about the file
print("Dataframe Info:")
print(df.info())
print("\nSample Data:")
print(df.head())
print("\nColumns:")
print(df.columns)

# Plot distributions for numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns
if numeric_columns.any():
    print("\nPlotting data distributions for numeric columns...\n")

    for column in numeric_columns:
        plt.figure(figsize=(10, 4))

        # Histogram
        plt.subplot(1, 2, 1)
        df[column].hist(bins=50, edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

        # Boxplot
        plt.subplot(1, 2, 2)
        df.boxplot(column=column, vert=False)
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)

        plt.tight_layout()
        plt.show()
else:
    print("No numeric columns found for plotting.")