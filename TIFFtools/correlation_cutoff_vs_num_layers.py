import xarray as xr
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Configuration for the file path and sampling parameters
CONFIG = {
    "file_path": '/Users/thowe/MinersAI Dropbox/Tyler Howe/ICB_data/testing/all_RS_layers.nc', #Path to nc file to analyze
    "samples": 500, #Number samples for the correlation matrix - impacts performance
    "target_count": 100, #Max number of lowly-correlated layers to group
    "max_attempts": 1000, #Max number of randomized attempts per threshold step
    "nan_threshold": 0.995, #Threshold percentage of pixels that are nan in the input layer
    "zero_threshold": 0.995, #Threshold percentage of pixels that are zero in the input layer
    "max_threshold": 1.0, #Max correlation threshold to test against
    "min_threshold": 0.0, #Min correlation threshold to test against
    "threshold_step": -0.02, #Threshold step for iterative analysis
}

def load_data(file_path: str, nan_threshold: float = 0.9, zero_threshold: float = 0.9) -> tuple:
    """
    Loads data from a NetCDF file, selecting layers with acceptable NaN and zero ratios.
    
    Parameters:
        file_path (str): Path to the NetCDF file.
        nan_threshold (float): Maximum acceptable ratio of NaN values per layer.
        zero_threshold (float): Maximum acceptable ratio of zero values per layer.
    
    Returns:
        tuple: A DataFrame of selected layers and a list of selected layer names.
    """
    print(f"Loading data from {file_path}...")
    try:
        ds = xr.open_dataset(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        return pd.DataFrame(), []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame(), []

    if "combined_layers" not in ds:
        print("Error: 'combined_layers' variable not found in the dataset.")
        return pd.DataFrame(), []

    combined_layers = ds["combined_layers"]

    # Check if dimensions are sufficient for sampling
    min_dimension = min(combined_layers.sizes['x'], combined_layers.sizes['y'])
    if min_dimension < int(np.sqrt(CONFIG["samples"])):
        print(f"Error: Minimum dimension ({min_dimension}) is smaller than the required square root of samples ({int(np.sqrt(CONFIG['samples']))}).")
        return pd.DataFrame(), []

    layer_names = combined_layers.layer.values

    # Check if the number of layers is less than target_count
    if len(layer_names) < CONFIG["target_count"]:
        print(f"Warning: Number of layers ({len(layer_names)}) is less than the target count ({CONFIG['target_count']}). Proceeding with available layers. \n")

    data = {}
    selected_layer_names = []
    
    for layer_name in layer_names:
        layer_data = combined_layers.sel(layer=layer_name).values.flatten()
        nan_ratio = np.isnan(layer_data).mean()
        zero_ratio = (layer_data == 0).mean()
        
        if nan_ratio < nan_threshold and zero_ratio < zero_threshold:
            data[layer_name] = layer_data
            selected_layer_names.append(layer_name)
        else:
            print(f"Layer '{layer_name}' excluded due to {nan_ratio:.2%} NaNs and {zero_ratio:.2%} zeros.")
    
    df = pd.DataFrame(data)
    return df, selected_layer_names

def calculate_correlation(df: pd.DataFrame, samples: int) -> pd.DataFrame:
    """
    Calculates the correlation matrix of sampled data from the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame of selected layer data.
        samples (int): Number of samples to include in the correlation calculation.
    
    Returns:
        pd.DataFrame: Correlation matrix of sampled data.
    """
    if samples > len(df):
        samples = len(df)
    df_sampled = df.sample(n=samples, random_state=42).fillna(0)
    
    if df_sampled.empty:
        print("Warning: DataFrame is empty, cannot calculate correlation.")
        return pd.DataFrame()
    
    return df_sampled.corr()

def select_random_lowly_correlated_layers(correlation_matrix: pd.DataFrame, threshold: float, target_count: int, max_attempts: int = 1000) -> list:
    """
    Selects a set of layers with low correlation based on a given threshold.
    
    Parameters:
        correlation_matrix (pd.DataFrame): Correlation matrix of layers.
        threshold (float): Correlation threshold for layer selection.
        target_count (int): Target number of layers to select.
        max_attempts (int): Maximum number of attempts to reach target_count.
    
    Returns:
        list: List of selected layer names meeting the threshold criteria.
    """
    if correlation_matrix.empty:
        print("Warning: Correlation matrix is empty, returning empty selection.")
        return []

    layers = list(correlation_matrix.columns)
    best_selection = []
    
    for attempt in range(max_attempts):
        random.shuffle(layers)
        selected_layers = []
        
        for layer in layers:
            if all(abs(correlation_matrix[layer][sel_layer]) < threshold for sel_layer in selected_layers):
                selected_layers.append(layer)
            if len(selected_layers) >= target_count:
                return selected_layers
        
        if len(selected_layers) > len(best_selection):
            best_selection = selected_layers
    
    return best_selection

def plot_results(results_df: pd.DataFrame):
    """
    Plots the number of selected layers against different correlation thresholds.
    
    Parameters:
        results_df (pd.DataFrame): DataFrame containing threshold values and the number of layers selected at each threshold.
    """
    if results_df.empty:
        print("Warning: No results to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Threshold'], results_df['Selected_Layers'], marker='o')
    plt.xlabel('Correlation Threshold')
    plt.xticks(np.arange(CONFIG["min_threshold"], CONFIG["max_threshold"] + (-CONFIG["threshold_step"]), -CONFIG["threshold_step"]), rotation=45)
    plt.ylabel('Number of Selected Layers')
    plt.title('Number of Layers Selected vs Correlation Threshold')
    plt.grid()
    plt.gca().invert_xaxis()
    plt.show()

def main():
    """
    Main function to load data, compute correlation matrix, and iteratively reduce correlation threshold to select layers.
    """
    df, layer_names = load_data(CONFIG["file_path"], CONFIG["nan_threshold"], CONFIG["zero_threshold"])
    
    if df.empty:
        print("Error: No valid data loaded, terminating process.")
        return
    
    correlation_matrix = calculate_correlation(df, CONFIG["samples"])
    
    results = []
    for threshold in np.arange(CONFIG["max_threshold"], CONFIG["min_threshold"], CONFIG["threshold_step"]):
        selected_layers = select_random_lowly_correlated_layers(correlation_matrix, threshold, CONFIG["target_count"], CONFIG["max_attempts"])
        results.append({'Threshold': threshold, 'Selected_Layers': len(selected_layers)})
        print(f"Threshold {threshold:.2f}: Selected {len(selected_layers)} layers")

    results_df = pd.DataFrame(results)


    plot_results(results_df)

if __name__ == "__main__":
    main()