import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree


def unsupervised_binary_classification(anomaly_scores, z_score: float = 3.5):  # Used to automatically classify the data points as inliers or outliers
    """
    Classifies data points as inliers or outliers using the Modified Z-Score method
    based on the Median Absolute Deviation (MAD).

    Outliers are identified as points with a modified Z-score greater than 3.5.

    Args:
        anomaly_scores (np.array): Computed anomaly scores
        z_score (float): Modified Z-score threshold for outlier classification (default: 3.5)

    Returns:
        np.array: Binary outlier classification (-1 outliers, 1 inliers)
    """
    median_score = np.median(anomaly_scores)
    mad = np.median(np.abs(anomaly_scores - median_score))
    modified_z_scores = 0.6745 * (anomaly_scores - median_score) / mad

    # Classify as outliers if modified z-score exceeds 3.5
    outlier_mask = np.abs(modified_z_scores) > z_score
    classifications = np.where(outlier_mask, -1, 1)

    return classifications


def isolation_forest(
    data: pd.DataFrame,
    feature_columns: list = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers in a geochemical dataset using the Isolation Forest algorithm.

    Parameters:
    - data (pd.DataFrame): The input geochemical dataset.
    - feature_columns (list): List of column names to be used as features.
                              If None, all numerical columns will be used.
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

    # Prepare the data
    features = data_copy[feature_columns].copy()

    # Handle missing values by imputing with column mean
    if features.isnull().values.any():
        features = features.fillna(features.mean())

    # Initialize and fit the Isolation Forest model
    iso_forest = IsolationForest(random_state=random_state)
    iso_forest.fit(features)

    # Compute decision function scores
    anomaly_scores = iso_forest.decision_function(features)

    # Use robust classification
    data_copy["outlier"] = unsupervised_binary_classification(anomaly_scores)
    data_copy["anomaly_score"] = anomaly_scores

    return data_copy


def local_outlier_factor(
    data: pd.DataFrame,
    feature_columns: list = None,
    n_neighbors: int = 20,
    scale_data: bool = False, 
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers in a geochemical dataset using the Local Outlier Factor (LOF) algorithm.

    Parameters:
    - data (pd.DataFrame): The input geochemical dataset.
    - feature_columns (list): List of column names to be used as features.
                              If None, all numerical columns will be used.
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
        n_neighbors=n_neighbors, novelty=False
    )
    lof.fit_predict(features)

    # Use negative outlier factor as anomaly scores
    anomaly_scores = lof.negative_outlier_factor_

    # Use robust classification
    data_copy["outlier"] = unsupervised_binary_classification(anomaly_scores)
    data_copy["anomaly_score"] = anomaly_scores

    return data_copy


def abod(
    data: pd.DataFrame,
    feature_columns: list = None,
    scale_data: bool = True,  # Default: True for better consistency
    use_knn: bool = False,
    k_neighbors: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optimized Angle-Based Outlier Detection (ABOD) for geochemical datasets.

    Parameters:
    - data (pd.DataFrame): Input dataset.
    - feature_columns (list): Columns to use for features (default: all numerical columns).
    - scale_data (bool): Whether to scale features using StandardScaler (default: True).
    - use_knn (bool): Use k-NN based ABOD instead of full dataset (default: False).
    - k_neighbors (int): Number of nearest neighbors to use for k-NN ABOD (default: 20).

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

    # Apply StandardScaler for consistent feature scaling
    if scale_data:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    n_samples = features.shape[0]
    abod_scores = np.zeros(n_samples)

    if use_knn:
        tree = KDTree(features)

    for query_index in range(n_samples):
        A = features[query_index]

        if use_knn:
            _, neighbor_indices = tree.query(A, k=k_neighbors + 1)
            neighbor_indices = neighbor_indices[1:]
        else:
            neighbor_indices = np.delete(np.arange(n_samples), query_index)

        T = features[neighbor_indices]
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
            abod_scores[query_index] = np.nan
        else:
            weighted_mean_squared = np.sum(weighted_squared_contributions) / weight_sum
            weighted_mean = np.sum(weighted_dot_products) / weight_sum
            abod_scores[query_index] = weighted_mean_squared - (weighted_mean**2)

        if np.isnan(abod_scores[query_index]):
            abod_scores[query_index] = np.nanmin(abod_scores[np.isfinite(abod_scores)])
        elif np.isinf(abod_scores[query_index]):
            abod_scores[query_index] = np.nanmin(abod_scores[np.isfinite(abod_scores)])

        print(
            f"Computed ABOD score for point {query_index + 1} of {n_samples}      {round((query_index/n_samples)*100)}%",
            end="\r",
        )

    abod_scores = np.nan_to_num(abod_scores, nan=np.nanmin(abod_scores))
    abod_scores[np.isinf(abod_scores)] = np.nanmin(abod_scores)

    abod_scores /= np.sqrt(n_samples) # Normalize scores based on number of samples

    # Ensure lower scores always indicate anomalies
    if np.mean(abod_scores) > 0:
        abod_scores *= -1

    # Store in DataFrame
    data_copy["anomaly_score"] = abod_scores
    data_copy["outlier"] = unsupervised_binary_classification(abod_scores)

    return data_copy
