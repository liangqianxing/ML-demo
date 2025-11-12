#!/usr/bin/env python3
"""Tiny deep-learning demo using PyTorch to classify synthetic clusters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Dataset:
    features: torch.Tensor
    labels: torch.Tensor


def generate_clusters(
    n_samples: int,
    n_classes: int,
    noise: float,
    seed: int,
) -> Dataset:
    """Generate simple blob data for multi-class classification."""
    if n_samples < n_classes:
        raise ValueError("n_samples must be >= n_classes.")

    generator = torch.Generator().manual_seed(seed)
    centers = torch.empty(n_classes, 2).uniform_(-6.0, 6.0, generator=generator)
    counts = torch.full((n_classes,), n_samples // n_classes, dtype=torch.long)
    counts[: n_samples % n_classes] += 1

    points = []
    labels = []
    for idx in range(n_classes):
        samples = centers[idx] + torch.randn(counts[idx], 2, generator=generator) * noise
        points.append(samples)
        labels.append(torch.full((counts[idx],), idx, dtype=torch.long))

    data = torch.cat(points, dim=0)
    target = torch.cat(labels, dim=0)
    return Dataset(features=data, labels=target)


class SimpleClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: list[int], num_classes: int):
        super().__init__()
        layers = []
        last = in_features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last, hidden))
            layers.append(nn.ReLU())
            last = hidden
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_test_split(dataset: Dataset, test_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset.features.shape[0], generator=generator)
    split = int(dataset.features.shape[0] * (1.0 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return (
        Dataset(features=dataset.features[train_idx], labels=dataset.labels[train_idx]),
        Dataset(features=dataset.features[test_idx], labels=dataset.labels[test_idx]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep learning demo with a small PyTorch MLP.")
    parser.add_argument("--samples", type=int, default=600, help="Total number of synthetic samples.")
    parser.add_argument("--classes", type=int, default=3, help="Number of classes.")
    parser.add_argument("--noise", type=float, default=1.0, help="Standard deviation of each cluster.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument("--hidden", type=int, nargs="*", default=(64, 64), help="Hidden layer sizes.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of data for testing.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. 'cpu' or 'cuda'.")
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    dataset = generate_clusters(args.samples, args.classes, args.noise, args.seed)
    train_data, test_data = train_test_split(dataset, args.test_ratio, args.seed)

    model = SimpleClassifier(in_features=2, hidden_sizes=list(args.hidden), num_classes=args.classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_x = train_data.features.to(device)
    train_y = train_data.labels.to(device)
    test_x = test_data.features.to(device)
    test_y = test_data.labels.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_acc = accuracy(logits, train_y)
                test_logits = model(test_x)
                test_acc = accuracy(test_logits, test_y)
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

    model.eval()
    with torch.no_grad():
        final_logits = model(test_x)
        final_acc = accuracy(final_logits, test_y)
        probabilities = final_logits.softmax(dim=1)

    print("\n=== Demo Summary ===")
    print(f"Device: {device} | Samples: {args.samples} | Classes: {args.classes}")
    print(f"Final test accuracy: {final_acc:.3f}")
    sample_indices = torch.arange(min(5, test_x.shape[0]))
    print("\nSample predictions (first few test points):")
    for idx in sample_indices:
        probs = probabilities[idx].cpu().numpy()
        pred = probs.argmax()
        print(f"Input: {test_x[idx].cpu().tolist()} -> Pred class {int(pred)} | Probabilities {probs.round(3)}")


if __name__ == "__main__":
    main()
