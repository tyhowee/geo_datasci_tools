import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from scipy.stats import f_oneway
from sklearn.metrics import mutual_info_score
from sklearn.metrics import roc_curve, auc


# Define plotting function for the outlier detection model results
def plot_outlier_results(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    score_col: str,
    binary_col: str,
    point_size: float = 100,
    score_title: str = "Anomaly Score",
    score_cbar_title: str = "Anomaly Score (lower = more anomalous)",
    binary_title: str = "Binary Classification",
    plot_title: str = "Outlier Detection Results",
    cmap: str = "viridis",  # Changed default to viridis
    binary_colors: dict = None,
):
    """
    Plot the results of outlier detection with a sequential colormap for anomaly scores
    and binary classification. Particularly suitable for geochemical data.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing geospatial and outlier information.
    - x_col (str): Column name for x-axis (e.g., 'Longitude').
    - y_col (str): Column name for y-axis (e.g., 'Latitude').
    - score_col (str): Column name for continuous anomaly scores.
    - binary_col (str): Column name for binary classification (-1 for outliers, 1 for inliers).
    - point_size (float): Size of the points in the scatter plots (default: 100).
    - score_title (str): Title for the score-based plot (default: "Anomaly Score").
    - binary_title (str): Title for the binary plot (default: "Binary Classification").
    - plot_title (str): Title for the entire figure (default: "Outlier Detection Results").
    - cmap (str): Colormap for the score plot (default: "viridis").
    - binary_colors (dict): Optional custom color map for binary classification
                            (default: {1: "blue", -1: "red"}).
    """
    # Default binary colors if not provided
    if binary_colors is None:
        binary_colors = {1: "#34495E", -1: "#D35400"}

    # Check if required columns exist in the data
    for col in [x_col, y_col, score_col, binary_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the data.")

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Plot the score-based heatmap
    sc1 = axes[0].scatter(
        data[x_col],
        data[y_col],
        c=data[score_col],
        cmap=cmap,
        s=point_size,
    )
    cbar1 = fig.colorbar(sc1, ax=axes[0])
    cbar1.set_label(score_cbar_title, fontsize=12)
    axes[0].set_title(score_title, fontsize=14)
    axes[0].set_xlabel(x_col, fontsize=12)
    axes[0].set_ylabel(y_col, fontsize=12)
    axes[0].grid(True)

    # Plot the binary classification
    axes[1].scatter(
        data[x_col],
        data[y_col],
        c=data[binary_col].map(binary_colors),
        s=point_size,
        label="Inliers/Outliers",
    )
    axes[1].set_title(binary_title, fontsize=14)
    axes[1].set_xlabel(x_col, fontsize=12)
    axes[1].set_ylabel(y_col, fontsize=12)
    axes[1].grid(True)

    # Add a legend for binary classification
    legend_labels = {
        1: "Inlier",
        -1: "Outlier",
    }
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10
        )
        for value, color in binary_colors.items()
    ]
    labels = [legend_labels.get(value, str(value)) for value in binary_colors.keys()]
    axes[1].legend(handles, labels, loc="upper right", fontsize=12)

    # Set the overall plot title
    plt.suptitle(plot_title, fontsize=18)
    plt.show()


# Define function to spatially plot validation results against all 3 model results
def plot_validation(
    outlier_datasets: list,
    outlier_dataset_names: list,
    validation_df: pd.DataFrame,
    x_col: str = "Longitude",
    y_col: str = "Latitude",
    binary_col: str = "outlier",
    point_size: float = 10,
    plot_title: str = "Outlier Detection Validation",
    colormap: str = "tab10",  # Changeable colormap
):

    # Get colormap and generate distinct colors
    cmap = plt.colormaps[colormap]
    colors = cmap(np.linspace(0, 1, len(outlier_datasets)))  # Generate distinct colors

    plt.figure(figsize=(10, 8))

    # Plot outlier datasets
    for i, (df, name) in enumerate(zip(outlier_datasets, outlier_dataset_names)):
        if binary_col not in df.columns:
            raise ValueError(f"Column '{binary_col}' not found in dataset {i+1}")

        # Filter only outliers (-1)
        outliers = df[df[binary_col] == -1]

        plt.scatter(
            outliers[x_col],
            outliers[y_col],
            c=[colors[i]],
            s=point_size,
            label=name,
            alpha=0.6,
            # edgecolor="black",
        )

    # Plot validation dataset as yellow stars
    plt.scatter(
        validation_df[x_col],
        validation_df[y_col],
        c="yellow",
        s=point_size * 30,  # Slightly larger for visibility
        marker="*",
        label="Known Mineral Occurrences",
        edgecolor="black",
    )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(plot_title, fontsize=14)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.grid(True)
    plt.show()


# Define function to plot bar charts of scores for multiple outlier detection methods
def plot_scores(
    score_dicts,  # List of output score dictionaries
    titles=None,  # Optional list of titles for each set
):

    num_sets = len(score_dicts)

    # Create figure with dynamic subplots if multiple sets exist
    _, axes = plt.subplots(1, num_sets, figsize=(6 * num_sets, 5), sharey=False)

    if num_sets == 1:
        axes = [axes]  # Ensure it's iterable

    if titles is None:
        titles = [f"Score Set {i+1}" for i in range(num_sets)]

    for ax, scores, title in zip(axes, score_dicts, titles):
        # Sort scores from highest to lowest
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        methods, values = zip(*sorted_scores) if sorted_scores else ([], [])

        # x positions
        x_positions = np.arange(len(methods))  # Use integer spacing

        # Plot histogram
        ax.bar(x_positions, values, alpha=0.7, width=0.5, align="center")

        # Set individual ylim with 10% headroom
        if values:
            ylim_max = max(values) * 1.1
        else:
            ylim_max = 1  # Default limit if no values

        ax.set_ylim(0, ylim_max)

        # Formatting
        # ax.set_xlabel("Outlier Detection Methods")
        # ax.set_ylabel("Score")  # Ensure all plots have y-labels
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


# Define function to plot ROC curves for multiple outlier detection methods
def plot_roc_curves(
    outlier_datasets,
    outlier_dataset_names,
    validation_df,
    x_col="Longitude",
    y_col="Latitude",
    prediction_col="anomaly_score",
    radius=0.01,
):
    """
    Computes and plots the ROC curves for multiple outlier detection methods
    based on spatial proximity to known mineral occurrences.

    Parameters:
    - outlier_datasets (list): List of DataFrames containing outlier data.
    - outlier_dataset_names (list): Corresponding list of dataset names.
    - validation_df (pd.DataFrame): DataFrame with known mineral occurrences.
    - x_col (str): Column name for longitude.
    - y_col (str): Column name for latitude.
    - prediction_col (str): Column name for anomaly scores (continuous values).
    - radius (float): Search radius (in degrees, roughly 1° ≈ 111 km at the equator).
    """

    plt.figure(figsize=(8, 6))  # Set figure size

    # Create a KD-Tree for fast spatial lookup of validation points
    validation_tree = cKDTree(validation_df[[x_col, y_col]].values)

    for df, name in zip(outlier_datasets, outlier_dataset_names):
        if prediction_col not in df.columns:
            raise ValueError(f"Column '{prediction_col}' not found in dataset '{name}'")

        # Query nearest validation points within the radius
        distances, _ = validation_tree.query(
            df[[x_col, y_col]].values, distance_upper_bound=radius
        )

        # Label as '1' (positive) if the outlier is close to a known deposit, else '0' (negative)
        df["is_near_deposit"] = (distances != np.inf).astype(int)

        # Ensure we have both positive and negative samples
        if len(df["is_near_deposit"].unique()) < 2:
            print(
                f"Warning: Only one class present in '{name}' dataset. Skipping ROC curve."
            )
            continue

        # Flip scores for all (lower scores indicate stronger anomalies)
        if name in ["IF", "LOF", "ABOD"]:
            df[prediction_col] = -df[prediction_col]  # Invert anomaly scores

        # Normalize scores to ensure consistency
        df[prediction_col] = (df[prediction_col] - df[prediction_col].min()) / (
            df[prediction_col].max() - df[prediction_col].min()
        )

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(df["is_near_deposit"], df[prediction_col])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    # Plot diagonal reference line (random classifier)
    plt.plot(
        [0, 1], [0, 1], color="gray", linestyle="--", lw=2, label="Random Classifier"
    )

    # Formatting
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Outlier Detection Models")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Show the plot
    plt.show()


# Define function to calculate ROC-AUC scores for outlier detection models
def calculate_roc_auc(
    outlier_datasets,
    outlier_dataset_names,
    validation_df,
    x_col="Longitude",
    y_col="Latitude",
    prediction_col="anomaly_score",  # Now using continuous scores
    radius=0.01,
):
    """
    Computes the ROC-AUC score for each outlier detection method based on spatial proximity to known mineral occurrences.

    Parameters:
    - outlier_datasets (list): List of DataFrames containing outlier data.
    - outlier_dataset_names (list): Corresponding list of dataset names.
    - validation_df (pd.DataFrame): DataFrame with known mineral occurrences.
    - x_col (str): Column name for longitude.
    - y_col (str): Column name for latitude.
    - prediction_col (str): Column name for continuous anomaly scores.
    - radius (float): Search radius (in degrees, roughly 1° ≈ 111 km at the equator).

    Returns:
    - A dictionary mapping dataset names to their respective spatial ROC-AUC scores.
    """

    roc_auc_scores = {}

    # Create a KD-Tree for fast spatial lookup of validation points
    validation_tree = cKDTree(validation_df[[x_col, y_col]].values)

    for df, name in zip(outlier_datasets, outlier_dataset_names):
        if prediction_col not in df.columns:
            raise ValueError(f"Column '{prediction_col}' not found in dataset '{name}'")

        # Query nearest validation points within the radius
        distances, _ = validation_tree.query(
            df[[x_col, y_col]].values, distance_upper_bound=radius
        )

        # Label as '1' (positive) if the outlier is close to a known deposit, else '0' (negative)
        df["is_near_deposit"] = (distances != np.inf).astype(int)

        # Ensure we have both positive and negative samples
        if len(df["is_near_deposit"].unique()) < 2:
            print(
                f"Warning: Only one class present in '{name}' dataset. Skipping ROC-AUC."
            )
            continue

        # Flip anomaly scores for all (since lower values indicate stronger anomalies)
        if name in ["IF", "LOF", "ABOD"]:
            df[prediction_col] = -df[
                prediction_col
            ]  # Invert anomaly scores **before normalization**

        # Normalize scores to ensure consistency across models
        min_val, max_val = df[prediction_col].min(), df[prediction_col].max()
        if max_val > min_val:  # Avoid division by zero
            df[prediction_col] = (df[prediction_col] - min_val) / (max_val - min_val)

        # Compute ROC-AUC score using continuous anomaly scores
        roc_auc = roc_auc_score(df["is_near_deposit"], df[prediction_col])
        roc_auc_scores[name] = roc_auc

        print(f"ROC-AUC Score for {name}: {roc_auc:.4f}")

    return roc_auc_scores


# Define function to calculate F-scores for outlier detection models
def calculate_f_score(
    outlier_datasets,
    outlier_dataset_names,
    validation_df,
    x_col="Longitude",
    y_col="Latitude",
    prediction_col="anomaly_score",  # Now using continuous predictions
    radius=0.01,
):
    """
    Computes the F-score (ANOVA F-statistic) for each outlier detection method based on spatial proximity to known mineral occurrences.

    Parameters:
    - outlier_datasets (list): List of DataFrames containing outlier data.
    - outlier_dataset_names (list): Corresponding list of dataset names.
    - validation_df (pd.DataFrame): DataFrame with known mineral occurrences.
    - x_col (str): Column name for longitude.
    - y_col (str): Column name for latitude.
    - prediction_col (str): Column name for continuous anomaly scores.
    - radius (float): Search radius (in degrees, roughly 1° ≈ 111 km at the equator).

    Returns:
    - A dictionary mapping dataset names to their respective spatial F-scores.
    """

    f_scores = {}

    # Create a KD-Tree for fast spatial lookup of validation points
    validation_tree = cKDTree(validation_df[[x_col, y_col]].values)

    for df, name in zip(outlier_datasets, outlier_dataset_names):
        if prediction_col not in df.columns:
            raise ValueError(f"Column '{prediction_col}' not found in dataset '{name}'")

        # Query nearest validation points within the radius
        distances, _ = validation_tree.query(
            df[[x_col, y_col]].values, distance_upper_bound=radius
        )

        # Label as '1' (positive) if the outlier is close to a known deposit, else '0' (negative)
        df["is_near_deposit"] = (distances != np.inf).astype(int)

        # Flip anomaly scores for all (since lower values indicate stronger anomalies)
        if name in ["IF", "LOF", "ABOD"]:
            df[prediction_col] = -df[
                prediction_col
            ]  # Flip scores **before** normalization

        # Normalize scores to ensure consistency across models
        min_val, max_val = df[prediction_col].min(), df[prediction_col].max()
        if max_val > min_val:  # Avoid division by zero
            df[prediction_col] = (df[prediction_col] - min_val) / (max_val - min_val)

        # Split into two groups based on spatial proximity
        group_near = df[df["is_near_deposit"] == 1][prediction_col]
        group_far = df[df["is_near_deposit"] == 0][prediction_col]

        # Check if we have enough data points to perform ANOVA
        if len(group_near) < 2 or len(group_far) < 2:
            print(
                f"Warning: Not enough data points in both groups for '{name}'. Skipping F-score calculation."
            )
            continue

        # Compute F-score (ANOVA F-statistic)
        f_stat, _ = f_oneway(group_near, group_far)
        f_scores[name] = f_stat

        print(f"F-Score for {name}: {f_stat:.4f}")

    return f_scores


# Define function to calculate Mutual Information scores for outlier detection models
def calculate_mi_score(
    outlier_datasets,
    outlier_dataset_names,
    validation_df,
    x_col="Longitude",
    y_col="Latitude",
    binary_col="outlier",
    radius=0.01,
):
    """
    Computes the Mutual Information (MI) score for each outlier detection method
    based on spatial proximity to known mineral occurrences.

    Parameters:
    - outlier_datasets (list): List of DataFrames containing outlier data.
    - outlier_dataset_names (list): Corresponding list of dataset names.
    - validation_df (pd.DataFrame): DataFrame with known mineral occurrences.
    - x_col (str): Column name for longitude.
    - y_col (str): Column name for latitude.
    - binary_col (str): Column name for binary classification (-1 for outliers, 1 for inliers).
    - radius (float): Search radius (in degrees, roughly 1° ≈ 111 km at the equator).

    Returns:
    - A dictionary mapping dataset names to their respective Mutual Information scores.
    """

    mi_scores = {}

    # Create a KD-Tree for fast spatial lookup of validation points
    validation_tree = cKDTree(validation_df[[x_col, y_col]].values)

    for df, name in zip(outlier_datasets, outlier_dataset_names):
        if binary_col not in df.columns:
            raise ValueError(f"Column '{binary_col}' not found in dataset '{name}'")

        # Convert outlier labels (-1 -> 1 for outliers, 1 -> 0 for inliers)
        df["predicted_outlier"] = np.where(df[binary_col] == -1, 1, 0)

        # Query nearest validation points within the radius
        distances, _ = validation_tree.query(
            df[[x_col, y_col]].values, distance_upper_bound=radius
        )

        # Label as '1' (positive) if the outlier is close to a known deposit, else '0' (negative)
        df["is_near_deposit"] = (distances != np.inf).astype(int)

        # Compute Mutual Information (MI) between predicted outlier status and validation proximity
        mi_score = mutual_info_score(df["predicted_outlier"], df["is_near_deposit"])
        mi_scores[name] = mi_score

        print(f"Mutual Information Score for {name}: {mi_score:.4f}")

    return mi_scores
