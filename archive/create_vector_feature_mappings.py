from tkinter import simpledialog, Tk
from tkinter.filedialog import askopenfilename
import geopandas as gpd
import json

def select_vector_file_and_feature():
    """
    Prompts the user to select a vector file and a specific feature within it, then outputs the feature mappings.
    """
    # Prompt the user to select a vector file
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes("-topmost", True)  # Bring the dialog to the front
    vector_file = askopenfilename(title="Select a Vector File", filetypes=[("GeoJSON files", "*.geojson"), ("All files", "*.*")])
    root.destroy()

    if not vector_file:
        print("No vector file selected.")
        return

    # Load the selected vector file
    gdf = gpd.read_file(vector_file)

    # Prompt the user to select a feature column within the file
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    feature_column = simpledialog.askstring("Select Feature", f"Enter the name of the feature column in {vector_file}:", initialvalue=gdf.columns[0])
    root.destroy()

    if not feature_column or feature_column not in gdf.columns:
        print("Invalid feature column selected.")
        return

    # Generate feature mappings
    unique_categories = gdf[feature_column].unique()
    feature_mappings = {cat: i for i, cat in enumerate(unique_categories)}

    # Convert mappings to JSON format and save to file
    mappings_json = json.dumps(feature_mappings, indent=4)
    output_file = vector_file.replace('.geojson', f'_{feature_column}_mappings.json')
    
    with open(output_file, "w") as f:
        f.write(mappings_json)

    print(f"Feature mappings for '{feature_column}' in '{vector_file}' saved to '{output_file}'.")

    return feature_mappings  # Return the mappings for immediate use, if needed

# Call the function to select a vector file and feature, then output mappings
feature_mappings = select_vector_file_and_feature()
