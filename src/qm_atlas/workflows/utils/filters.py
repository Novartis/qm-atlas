#!/usr/bin/env python

"""
Filtering and clustering functions for groups of molecules
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_scaled_kmeans(n_clusters: int = 3, random_seed: int = 42) -> Pipeline:
    """Create a pipeline with scaling and KMeans clustering.

    Args:
        n_clusters: Number of clusters for KMeans
        random_seed: Random seed for reproducibility

    Returns:
        Pipeline: sklearn Pipeline with StandardScaler and KMeans models
    """
    scaled_kmeans = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)),
        ]
    )
    return scaled_kmeans


def cluster_kmeans(
    values: np.ndarray, n_clusters: int = 3, random_seed: int = 42
) -> list[list[int]]:
    """Cluster values using KMeans with scaling.

    Args:
        values: Input array to cluster
        n_clusters: Number of clusters
        random_seed: Random seed for reproducibility

    Returns:
        List[List[Int]]: Groups of indices for each cluster
    """
    # reduce number of clusters if there are fewer points than cluster centers
    n_clusters = min([n_clusters, values.shape[0]])

    scaled_kmeans = get_scaled_kmeans(n_clusters=n_clusters, random_seed=random_seed)
    # Pipeline.fit_predict() returns ndarray of cluster labels with shape (N,)
    # containing the index of the cluster each sample belongs to
    labels = scaled_kmeans.fit_predict(values)

    label_to_idcs = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        label_to_idcs[int(label)].append(idx)

    return label_to_idcs


def group_kmeans(
    values: np.ndarray, n_clusters: int = 5, random_seed: int = 66
) -> list[np.ndarray]:
    """Perform KMeans clustering on values.

    Args:
        values: NxK matrix, where N is len of properties and K is len of datapoints
        n_clusters: Number of clusters
        random_seed: Random seed for reproducibility

    Returns:
        List[ndarray]: Groups of indices for each cluster
    """

    values = values.T

    if values.shape[0] == 0:
        return []

    values = StandardScaler().fit_transform(values)

    # KMeans.fit() returns a fitted KMeans object; .labels_ is ndarray of cluster assignments
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10).fit(values)

    labels = kmeans.labels_

    cluster_labels, _ = np.unique(labels, return_counts=True)

    cluster_indices = []

    for label in cluster_labels:
        (idxs,) = np.where(labels == label)
        cluster_indices.append(idxs)

    return cluster_indices


def group_dbscan(
    values: np.ndarray, min_samples: int = 2, eps: float = 0.2
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Perform DBSCAN clustering on values.

    Args:
        values: NxK matrix, where N is len of properties and K is len of datapoints
        min_samples: Minimum samples in a neighborhood to form a core point
        eps: Maximum distance between two samples for DBSCAN

    Returns:
        Tuple containing:
        - List[ndarray]: Groups of indices for each cluster
        - ndarray: Outlier indices
    """

    values = values.T

    if values.shape[0] == 0:
        return [], np.array([])

    values = StandardScaler().fit_transform(values)

    # DBSCAN.fit() returns a fitted DBSCAN object; .labels_ is ndarray with cluster assignments (-1 for outliers)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(values)
    labels = db.labels_
    cluster_labels, _ = np.unique(labels, return_counts=True)

    cluster_indices = []
    outliers = np.array([], dtype=np.int64)
    for label in cluster_labels:
        (idxs,) = np.where(labels == label)

        if label == -1:
            outliers = idxs
        else:
            cluster_indices.append(idxs)

    return cluster_indices, np.asarray(outliers, dtype=np.int64)


def group_isolationforest(
    values: np.ndarray,
    random_seed: int = 67,
    max_samples: int = 100,
    use_standardscale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Isolation Forest anomaly detection on values.

    Args:
        values: NxK matrix, where N is len of properties and K is len of datapoints
        random_seed: Random seed for reproducibility
        max_samples: Number of samples to draw for training
        use_standardscale: Whether to apply StandardScaler before fitting

    Returns:
        Tuple containing:
        - ndarray: Outlier indices
        - ndarray: Inlier indices
    """

    values = values.T

    if values.shape[0] == 0:
        return np.array([]), np.array([])

    if use_standardscale:
        values = StandardScaler().fit_transform(values)

    # IsolationForest.fit() returns a fitted IsolationForest object
    # IsolationForest.predict() returns ndarray with predictions (1 for inliers, -1 for outliers)
    clf = IsolationForest(max_samples=max_samples, random_state=random_seed)
    clf.fit(values)

    predictions = clf.predict(values)

    outlier_label = -1
    (idxs,) = np.where(predictions == outlier_label)
    idxs = idxs.flatten()

    label = 1
    (nidxs,) = np.where(predictions == label)
    nidxs = nidxs.flatten()

    return idxs, nidxs


def combination(
    values: np.ndarray,
    n_clusters: int = 5,
    random_seed: int = 67,
    eps: float = 0.2,
    max_samples: int = 100,
    return_splits: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Combine multiple clustering/anomaly detection methods.

    Args:
        values: NxK matrix, where N is len of properties and K is len of datapoints
        n_clusters: Number of clusters for KMeans
        random_seed: Random seed for reproducibility
        eps: DBSCAN eps parameter
        max_samples: IsolationForest max_samples parameter
        return_splits: If True, return detailed splits; otherwise return combined groups

    Returns:
        List[ndarray] of groups, or if return_splits=True, tuple of (groups, outlier_groups, outliers)
    """

    n_values = len(values[0])

    if n_values == 1:
        return [np.array([0])]

    if n_values < max_samples:
        max_samples = n_values

    outlier_groups = []
    groups = []

    outliers, notoutliers = group_isolationforest(
        values, max_samples=max_samples, random_seed=random_seed
    )

    values_outliers = values[:, outliers]

    groups_dbscan, outliers_dbscan = group_dbscan(values_outliers, eps=eps)

    for group in groups_dbscan:
        idxs = outliers[group]
        outlier_groups.append(idxs)

    # Map DBSCAN outliers back to original indices only if there are outliers
    if len(outliers_dbscan) > 0:
        outliers_dbscan = outliers[outliers_dbscan]
    else:
        outliers_dbscan = np.array([], dtype=np.int64)

    # Crashes when too few conformer < n_clusters
    n_clusters = min(n_clusters, len(notoutliers))

    groups_kmeans = group_kmeans(values[:, notoutliers], n_clusters=n_clusters)

    for group in groups_kmeans:
        idxs = notoutliers[group]
        groups.append(idxs)

    if return_splits:
        return groups, outlier_groups, outliers_dbscan

    outliers_dbscan = [[k] for k in outliers_dbscan]

    return groups + outlier_groups + outliers_dbscan


def pick_n_from_groups(
    cluster_indices: list[np.ndarray], n_picks: int, outliers: np.ndarray | None = None
) -> list[int]:
    """
    Pick N points from each group.

    Args:
        cluster_indices: List of arrays containing indices for each cluster
        n_picks: Number of points to pick from each group
        outliers: Optional array of outlier indices to include

    Returns:
        List[Int]: Flattened list of selected indices
    """

    filtered_indices = [idxs[:n_picks] for idxs in cluster_indices]
    filtered_indices = [item for sublist in filtered_indices for item in sublist]

    if outliers is not None:
        filtered_indices += list(outliers)

    return filtered_indices
