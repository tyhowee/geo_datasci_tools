import pandas as pd

# Replace 'your_file.csv' with your actual file path
file_path = '/Users/thowe/Downloads/USA_Idaho_Cobalt_Belt_Radiometric.csv'

# Read only the header of the CSV
columns = pd.read_csv(file_path, nrows=0).columns

# Print column names
print(columns.tolist())