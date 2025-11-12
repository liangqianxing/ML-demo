#!/usr/bin/env python3
"""Minimal linear-regression demo using only NumPy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class LinearRegressionModel:
    """Holds intercept and slope for a single-feature linear regression."""

    intercept: float
    coef: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float).reshape(-1)
        return self.intercept + self.coef * values


def generate_dataset(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible synthetic dataset y = 3.5 * x + 12 + noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, size=n_samples)
    y = 3.5 * x + 12.0 + rng.normal(0.0, noise, size=n_samples)
    return x, y


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> LinearRegressionModel:
    """Solve the normal equation analytically since this is a toy example."""
    ones = np.ones_like(x)
    design_matrix = np.column_stack([ones, x])
    theta = np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T @ y
    return LinearRegressionModel(intercept=float(theta[0]), coef=float(theta[1]))


def evaluate(model: LinearRegressionModel, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    predictions = model.predict(x)
    residuals = y - predictions
    mse = np.mean(residuals ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum(residuals ** 2) / total
    return mse, r2


def train_test_split(x: np.ndarray, y: np.ndarray, test_ratio: float, seed: int) -> tuple[np.ndarray, ...]:
    if x.shape[0] < 2:
        raise ValueError("Need at least two samples for a train/test split.")
    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    test_size = max(1, min(x.shape[0] - 1, int(x.shape[0] * test_ratio)))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple linear regression demo.")
    parser.add_argument("--samples", type=int, default=80, help="Number of generated samples.")
    parser.add_argument("--noise", type=float, default=4.0, help="Standard deviation of target noise.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--predict",
        type=float,
        nargs="*",
        help="Optional values to predict once the model is trained.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y = generate_dataset(args.samples, args.noise, args.seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.2, seed=args.seed)

    model = fit_linear_regression(x_train, y_train)
    train_mse, train_r2 = evaluate(model, x_train, y_train)
    test_mse, test_r2 = evaluate(model, x_test, y_test)

    print("=== Linear Regression Demo ===")
    print(f"Learned model: y = {model.intercept:.3f} + {model.coef:.3f} * x")
    print(f"Train MSE: {train_mse:.3f} | Train R^2: {train_r2:.3f}")
    print(f" Test MSE: {test_mse:.3f} |  Test R^2: {test_r2:.3f}")

    if args.predict:
        preds = model.predict(np.array(args.predict))
        print("\nPredictions for custom inputs:")
        for value, pred in zip(args.predict, preds):
            print(f"x = {value:6.2f} -> y_hat = {pred:7.3f}")


if __name__ == "__main__":
    main()
