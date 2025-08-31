import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from typing import Tuple, List


def initialize_centroids(X: np.ndarray, k: int) -> np.ndarray:
    """
    Randomly selects k data points from the dataset as initial centroids.

    Args:
        X: The input data of shape (n_samples, n_features).
        k: The number of clusters.

    Returns:
        An array of shape (k, n_features) representing the initial centroids.
    """
    n_samples = X.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    return centroids


def assign_to_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assigns each data point to the closest centroid.

    Args:
        X: The input data of shape (n_samples, n_features).
        centroids: The current centroids of shape (k, n_features).

    Returns:
        An array of shape (n_samples,) with cluster indices for each data point.
    """
    # Calculate the squared Euclidean distance from each point to each centroid.
    # The use of np.newaxis triggers broadcasting to compute distances efficiently.
    # The resulting shape is (k, n_samples).
    squared_distances = ((X - centroids[:, np.newaxis]) ** 2).sum(axis=2)

    # Find the index of the closest centroid for each data point.
    # np.argmin along axis=0 finds the minimum in each column (i.e., for each sample).
    # Taking the sqrt is not necessary as argmin(d^2) == argmin(d).
    return np.argmin(squared_distances, axis=0)


def update_centroids(
    X: np.ndarray, labels: np.ndarray, k: int, old_centroids: np.ndarray
) -> np.ndarray:
    """
    Updates the centroids by calculating the mean of all points in each cluster.

    Args:
        X: The input data of shape (n_samples, n_features).
        labels: The cluster assignments for each data point.
        k: The number of clusters.
        old_centroids: The centroids from the previous iteration, used for empty clusters.

    Returns:
        An array of new centroids of shape (k, n_features).
    """
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0)
        else:
            # If a cluster becomes empty, keep its old centroid position
            # to prevent it from vanishing.
            new_centroids[i] = old_centroids[i]
    return new_centroids


def k_means(
    X: np.ndarray, k: int, max_iters: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs the K-means clustering algorithm.

    Args:
        X: The input data of shape (n_samples, n_features).
        k: The number of clusters.
        max_iters: The maximum number of iterations to run.

    Returns:
        A tuple containing:
        - The final centroids.
        - The final cluster labels for each data point.
    """
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_to_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, centroids)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels


def main():
    """
    Main function to generate data, run K-means, and plot the results.
    """
    # 1. Generate synthetic data for testing
    # We create 300 samples with 4 distinct clusters.
    n_samples = 300
    n_features = 2
    n_clusters = 3
    random_state = 42

    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.9,
        random_state=random_state,
    )

    # 2. Run the K-means algorithm
    k = 4
    centroids, labels = k_means(X, k)

    # 3. Plot the results for visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis", alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, alpha=0.9, marker="X")
    plt.title("K-means Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


if __name__ == "__main__":
    main()
