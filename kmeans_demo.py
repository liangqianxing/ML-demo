#!/usr/bin/env python3
"""Simple K-Means clustering demo that depends only on NumPy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class KMeansResult:
    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    iterations: int


def generate_blobs(
    n_samples: int,
    n_clusters: int,
    n_features: int,
    cluster_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic multi-cluster data for clustering practice."""
    rng = np.random.default_rng(seed)
    base_centers = rng.uniform(-10.0, 10.0, size=(n_clusters, n_features))
    counts = np.full(n_clusters, n_samples // n_clusters, dtype=int)
    counts[: n_samples % n_clusters] += 1

    samples = []
    labels = []
    for idx, center in enumerate(base_centers):
        if counts[idx] == 0:
            continue
        scale = cluster_std * rng.uniform(0.8, 1.2)
        cov = np.eye(n_features) * scale
        points = rng.multivariate_normal(center, cov, size=counts[idx])
        samples.append(points)
        labels.append(np.full(points.shape[0], idx))

    data = np.vstack(samples)
    label_array = np.concatenate(labels)
    return data, label_array


def kmeans(
    data: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> KMeansResult:
    """Basic Lloyd's algorithm implementation."""
    rng = np.random.default_rng(seed)
    initial_idx = rng.choice(data.shape[0], size=k, replace=False)
    centroids = data[initial_idx].copy()
    labels = np.zeros(data.shape[0], dtype=int)

    for iteration in range(1, max_iter + 1):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()
        for cluster_id in range(k):
            members = data[labels == cluster_id]
            if members.size == 0:
                # Re-initialize empty clusters to a random point.
                new_centroids[cluster_id] = data[rng.integers(0, data.shape[0])]
            else:
                new_centroids[cluster_id] = members.mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift <= tol:
            break

    inertia = np.sum((data - centroids[labels]) ** 2)
    return KMeansResult(centroids=centroids, labels=labels, inertia=float(inertia), iterations=iteration)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised learning demo via K-Means clustering.")
    parser.add_argument("--samples", type=int, default=300, help="Number of generated samples.")
    parser.add_argument("--clusters", type=int, default=4, help="Number of clusters to generate and fit.")
    parser.add_argument("--features", type=int, default=2, help="Number of features per sample.")
    parser.add_argument("--cluster-std", type=float, default=1.2, help="Standard deviation of each blob.")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum number of K-Means iterations.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Convergence tolerance on centroid shift.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, true_labels = generate_blobs(
        n_samples=args.samples,
        n_clusters=args.clusters,
        n_features=args.features,
        cluster_std=args.cluster_std,
        seed=args.seed,
    )

    result = kmeans(
        data=data,
        k=args.clusters,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
    )

    print("=== K-Means Demo ===")
    print(f"Samples: {args.samples} | Features: {args.features} | Target clusters: {args.clusters}")
    print(f"Converged in {result.iterations} iterations with inertia {result.inertia:.2f}")
    print("\nCentroids (each row = cluster center):")
    np.set_printoptions(precision=3, suppress=True)
    print(result.centroids)

    counts = np.bincount(result.labels, minlength=args.clusters)
    print("\nCluster sizes:")
    for idx, size in enumerate(counts):
        print(f"Cluster {idx:>2}: {size:4d} points")

    # Compare learned assignments with original synthetic labels to show performance.
    # Because K-Means label order is arbitrary, just compute purity-style score.
    contingency = np.zeros((args.clusters, args.clusters), dtype=int)
    for pred, true in zip(result.labels, true_labels):
        contingency[pred, true] += 1
    matches = contingency.max(axis=1).sum()
    print(f"\nApproximate clustering accuracy: {matches / args.samples:.3f} (order-invariant)")


if __name__ == "__main__":
    main()
