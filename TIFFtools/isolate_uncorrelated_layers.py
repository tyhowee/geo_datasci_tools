import xarray as xr
import pandas as pd
import numpy as np
import random
import os
import shutil


CONFIG = {
    "file_path": '/Users/thowe/MinersAI Dropbox/Tyler Howe/ICB_data/cube_testing/all_RS_layers.nc', #Path to nc file containing all layers of interest
    "samples": 500, #For corr(), impacts performance
    "nan_threshold": 0.995, #Max percentage of pixels in any layer that can be nan
    "zero_threshold": 0.8, #Max percentage of pixels in any layer that can be zero
    "correlation_threshold": 1,  # Maximum allowed correlation between any pair of selected layers
    "target_count": 10,  # Number of layers to select
    "max_attempts": 1000,  # Maximum number of attempts to try different random selections
    "folder_path": '/Users/thowe/MinersAI Dropbox/Tyler Howe/ICB_data/geospatial_data/all_RS_layer', #Folder containing files for all layers present in nc file
    "output_folder": '/Users/thowe/MinersAI Dropbox/Tyler Howe/ICB_data/geospatial_data/uncorrelated_<0.8_zero' #Output folder to export lowly-correlated files
}

def load_data(file_path: str, nan_threshold: float = 0.9, zero_threshold: float = 0.9) -> pd.DataFrame:
    """
    Loads the NetCDF file, reshapes each layer into a DataFrame, and excludes layers with NaN or zero values
    exceeding specified thresholds.
    
    Args:
        file_path (str): Path to the NetCDF file.
        nan_threshold (float): Maximum proportion of NaNs allowed for a layer to be included.
        zero_threshold (float): Maximum proportion of zero values allowed for a layer to be included.

    Returns:
        pd.DataFrame: DataFrame of the selected layers.
        list: List of layer names that meet the NaN and zero value threshold criteria.

    Raises:
        FileNotFoundError: If the file at `file_path` does not exist.
        ValueError: If `nan_threshold` or `zero_threshold` is not between 0 and 1.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    if not (0 <= nan_threshold <= 1) or not (0 <= zero_threshold <= 1):
        raise ValueError("Thresholds must be between 0 and 1.")

    ds = xr.open_dataset(file_path)
    combined_layers = ds["combined_layers"]
    layer_names = combined_layers.layer.values  # Assumes layer names are stored in the 'layer' dimension

    data = {}
    selected_layer_names = []
    for layer_name in layer_names:
        layer_data = combined_layers.sel(layer=layer_name).values.flatten()
        nan_ratio = np.isnan(layer_data).mean()  # Proportion of NaNs
        zero_ratio = (layer_data == 0).mean()     # Proportion of zeros
        
        if nan_ratio < nan_threshold and zero_ratio < zero_threshold:
            data[layer_name] = layer_data
            selected_layer_names.append(layer_name)
        else:
            print(f"Layer '{layer_name}' excluded due to {nan_ratio:.2%} NaNs and {zero_ratio:.2%} zeros.")

    df = pd.DataFrame(data)
    return df, selected_layer_names


def calculate_correlation(df: pd.DataFrame, samples: int) -> pd.DataFrame:
    """
    Downsamples the DataFrame and calculates the correlation matrix, handling NaN values.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        samples (int): Number of samples to use for correlation calculation.

    Returns:
        pd.DataFrame: Correlation matrix of the sampled DataFrame.

    Raises:
        ValueError: If `samples` is less than 1.
    """
    if samples < 1:
        raise ValueError("Samples must be at least 1.")
    if samples > len(df):
        samples = len(df)
    
    df_sampled = df.sample(n=samples, random_state=42).fillna(0)
    correlation_matrix = df_sampled.corr()
    return correlation_matrix


def select_random_lowly_correlated_layers(correlation_matrix: pd.DataFrame, threshold: float = 0.3, target_count: int = 10, max_attempts: int = 1000) -> list:
    """
    Randomly selects a set of layers with pairwise correlations below the threshold.
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix.
        threshold (float): Maximum allowed correlation between any pair of selected layers.
        target_count (int): Number of layers to select.
        max_attempts (int): Maximum number of attempts to try different random selections.

    Returns:
        list: A list of selected layer names or the highest number of uncorrelated layers found.

    Raises:
        ValueError: If `threshold` is not between 0 and 1 or if `target_count` or `max_attempts` is less than 1.
    """
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")
    if target_count < 1 or max_attempts < 1:
        raise ValueError("Target count and max attempts must be at least 1.")
    
    layers = list(correlation_matrix.columns)
    best_selection = []
    
    for attempt in range(max_attempts):
        random.shuffle(layers)
        selected_layers = []

        for layer in layers:
            if all(abs(correlation_matrix[layer][sel_layer]) < threshold for sel_layer in selected_layers):
                selected_layers.append(layer)
            
            if len(selected_layers) >= target_count:
                print(f"Found a set of {target_count} layers on attempt {attempt + 1}")
                return selected_layers
        
        if len(selected_layers) > len(best_selection):
            best_selection = selected_layers
            print(f"New best selection with {len(best_selection)} layers on attempt {attempt + 1}")
        
        if (attempt + 1) % 10 == 0:
            print(f"Attempt {attempt + 1}: Current best selection has {len(best_selection)} layers")

    print(f"Unable to find a set of {target_count} layers with pairwise correlations below {threshold} after {max_attempts} attempts.")
    return best_selection


def select_files(folder_path: str, selected_layers: list, output_folder: str):
    """
    Copies files matching selected layer names from the input folder to an output folder.
    
    Args:
        folder_path (str): Path to the folder containing the files.
        selected_layers (list): List of layer names to select (without extensions).
        output_folder (str): Path to the output folder where selected files will be copied.

    Raises:
        FileNotFoundError: If `folder_path` does not exist.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    os.makedirs(output_folder, exist_ok=True)
    
    selected_layers_set = set(selected_layers)
    
    for filename in os.listdir(folder_path):
        name_without_extension = os.path.splitext(filename)[0]
        
        if name_without_extension in selected_layers_set:
            src_path = os.path.join(folder_path, filename)
            dst_path = os.path.join(output_folder, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename}")


def main():
    try:
        df, layer_names = load_data(CONFIG["file_path"], nan_threshold=CONFIG["nan_threshold"], zero_threshold=CONFIG["zero_threshold"])
        correlation_matrix = calculate_correlation(df, CONFIG["samples"])
        selected_layers = select_random_lowly_correlated_layers(correlation_matrix, CONFIG["correlation_threshold"], CONFIG["target_count"], CONFIG["max_attempts"])
        
        if len(selected_layers) < CONFIG["target_count"]:
            print(f"\nOnly {len(selected_layers)} layers found that meet the pairwise correlation criterion.")
        else:
            print(f"\nSelected {CONFIG['target_count']} layers with pairwise correlations below {CONFIG['correlation_threshold']}:")
        print(selected_layers)

        select_files(CONFIG["folder_path"], selected_layers, CONFIG["output_folder"])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()