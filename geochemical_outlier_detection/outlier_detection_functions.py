import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def isolation_forest(
    data: pd.DataFrame,
    feature_columns: list = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers in a geochemical dataset using the Isolation Forest algorithm.

    Parameters:
    - data (pd.DataFrame): The input geochemical dataset.
    - feature_columns (list): List of column names to be used as features.
                              If None, all numerical columns will be used.
    - contamination (float): The proportion of outliers in the dataset (default: 0.05).
    - random_state (int): Random seed for reproducibility (default: 42).

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]:
        1. DataFrame with additional columns:
            - 'outlier': Binary classification (-1 = outlier, 1 = inlier).
            - 'anomaly_score': Continuous anomaly score (lower = more anomalous).
        2. DataFrame containing only the detected outliers.
    """
    # Make a copy of the input DataFrame to avoid modifying the original
    data_copy = data.copy()

    # Validate inputs
    if not isinstance(data_copy, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # Use all numerical columns if feature_columns is None
    if feature_columns is None:
        feature_columns = data_copy.select_dtypes(include=[np.number]).columns.tolist()
        if not feature_columns:
            raise ValueError(
                "No numerical columns found in the dataset to use as features."
            )

    if not all(col in data_copy.columns for col in feature_columns):
        raise ValueError("Some feature columns are not present in the dataset.")
    if not 0 < contamination < 0.5:
        raise ValueError("Contamination must be between 0 and 0.5.")

    # Prepare the data
    features = data_copy[feature_columns].copy()

    # Handle missing values by imputing with column mean
    if features.isnull().values.any():
        features = features.fillna(features.mean())

    # Initialize and fit the Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    iso_forest.fit(features)

    # Add the binary classification to the copied DataFrame
    data_copy["outlier"] = iso_forest.predict(features)

    # Add the continuous anomaly scores to the copied DataFrame
    data_copy["anomaly_score"] = iso_forest.decision_function(features)

    return data_copy


def local_outlier_factor(
    data: pd.DataFrame,
    feature_columns: list = None,
    contamination: float = 0.05,
    n_neighbors: int = 20,
    scale_data: bool = False,  # New argument to enable/disable scaling
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers in a geochemical dataset using the Local Outlier Factor (LOF) algorithm.

    Parameters:
    - data (pd.DataFrame): The input geochemical dataset.
    - feature_columns (list): List of column names to be used as features.
                              If None, all numerical columns will be used.
    - contamination (float): The proportion of outliers in the dataset (default: 0.05).
    - n_neighbors (int): Number of neighbors to use for LOF (default: 20).
    - scale_data (bool): Whether to scale features using StandardScaler (default: False).

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]:
        1. DataFrame with additional columns:
            - 'outlier': Binary classification (-1 = outlier, 1 = inlier).
            - 'anomaly_score': Continuous anomaly score (lower = more anomalous).
        2. DataFrame containing only the detected outliers.
    """

    data_copy = data.copy()

    # Validate inputs
    if not isinstance(data_copy, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # Use all numerical columns if feature_columns is None
    if feature_columns is None:
        feature_columns = data_copy.select_dtypes(include=[np.number]).columns.tolist()
        if not feature_columns:
            raise ValueError(
                "No numerical columns found in the dataset to use as features."
            )

    if not all(col in data_copy.columns for col in feature_columns):
        raise ValueError("Some feature columns are not present in the dataset.")
    if not 0 < contamination < 0.5:
        raise ValueError("Contamination must be between 0 and 0.5.")

    # Prepare the data
    features = data_copy[feature_columns].copy()

    # Handle missing values by imputing with column mean
    features = features.fillna(features.mean())

    # Apply feature scaling if enabled
    if scale_data:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    # Initialize and fit LOF model
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, novelty=False
    )
    data_copy["outlier"] = lof.fit_predict(features)
    data_copy["anomaly_score"] = lof.negative_outlier_factor_

    return data_copy


def abod(
    data: pd.DataFrame,
    feature_columns: list = None,
    scale_data: bool = False,
    contamination: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optimized Angle-Based Outlier Detection (ABOD) for geochemical datasets.

    Parameters:
    - data (pd.DataFrame): Input dataset.
    - feature_columns (list): Columns to use for features (default: all numerical columns).
    - scale_data (bool): Whether to scale features using StandardScaler (default: False).
    - contamination (float): The proportion of outliers in the dataset (default: 0.05).

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]:
        1. DataFrame with additional columns for outlier classification and anomaly score
        2. DataFrame containing only the detected outliers
    """
    data_copy = data.copy()

    if feature_columns is None:
        feature_columns = data_copy.select_dtypes(include=[np.number]).columns.tolist()
        if not feature_columns:
            raise ValueError("No numerical columns found to use as features.")

    features = (
        data_copy[feature_columns].fillna(data_copy[feature_columns].median()).values
    )

    if scale_data:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    n_samples = features.shape[0]
    abod_scores = np.zeros(n_samples)

    for query_index in range(n_samples):
        A = features[query_index]
        T = np.delete(features, query_index, axis=0)
        vectors = T - A
        norms = np.linalg.norm(vectors, axis=1)
        valid_indices = norms > 0
        vectors = vectors[valid_indices]
        norms = norms[valid_indices]

        unit_vectors = vectors / norms[:, None]
        dot_products = np.dot(unit_vectors, unit_vectors.T)
        norm_products = np.outer(norms, norms)
        weights = 1 / norm_products

        weighted_dot_products = dot_products * weights
        weighted_squared_contributions = weighted_dot_products**2

        weight_sum = np.sum(weights)
        if weight_sum == 0:
            abod_scores[query_index] = float("inf")
        else:
            weighted_mean_squared = np.sum(weighted_squared_contributions) / weight_sum
            weighted_mean = np.sum(weighted_dot_products) / weight_sum
            abod_scores[query_index] = weighted_mean_squared - (weighted_mean**2)

        print(
            f"Computed ABOD score for point {query_index + 1} of {n_samples}      {round((query_index/n_samples)*100)}%",
            end="\r",
        )

    data_copy["anomaly_score"] = abod_scores

    # Determine threshold based on contamination
    num_outliers = int(contamination * n_samples)  # Determine number of outliers
    threshold = np.partition(abod_scores, num_outliers)[num_outliers]  # Find score threshold

    # Assign binary classification (-1 for outliers, 1 for inliers)
    data_copy["outlier"] = np.where(abod_scores <= threshold, -1, 1)

    return data_copy
