from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F


def add_gaussian_noise(embeddings, mean=0.0, std=0.01):
    noise = torch.normal(mean=mean, std=std, size=embeddings.shape)
    return embeddings + noise


def mixup_embeddings(embeddings, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(embeddings.size(0))
    mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[idx]
    return mixed_embeddings


class BaseSampler(ABC):
    @abstractmethod
    def sample(self, buffer, n_samples, **kwargs):
        pass


class AllSampling(BaseSampler):
    def sample(self, buffer, n_samples, **kwargs):
        x_memory, y_memory = [], []

        for samples in buffer.values():
            if len(samples) > 0:
                x_memory.extend([x for x, _ in samples])
                y_memory.extend([y for _, y in samples])

        return torch.stack(x_memory), torch.tensor(y_memory, dtype=torch.long)


class RandomSampling:
    def sample(self, embeddings: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Efficient random sampling for single class"""
        if embeddings.size(0) <= n_samples:
            return embeddings
            
        indices = torch.randperm(embeddings.size(0))[:n_samples]
        return embeddings[indices]


class EntropySampling(BaseSampler):
    def _compute_entropy(self, predictions: torch.Tensor) -> np.ndarray:
        pred_np = predictions.detach().cpu().numpy()

        entropies = entropy(pred_np, axis=1)

        return entropies

    def sample(self, buffer, n_samples, **kwargs):
        get_predictions = kwargs.get("get_predictions")
        device = kwargs.get("device")
        new_buffer = defaultdict(list)

        for label, samples in buffer.items():
            if len(samples) <= n_samples:
                new_buffer[label] = samples
                continue

            predictions = F.softmax(
                get_predictions(torch.stack(samples).to(device)), dim=1
            )

            entropies = self._compute_entropy(predictions)

            top_indices = np.argsort(entropies)[-n_samples:]
            new_buffer[label] = [samples[i] for i in top_indices]

        return new_buffer


class BoundarySampling(BaseSampler):
    def _compute_boundary_scores(
        self, X: np.ndarray, labels: np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(X)

        distances, indices = nn.kneighbors(X)

        boundary_scores = np.mean(labels[indices] != labels[:, np.newaxis], axis=1)

        return boundary_scores

    def sample(self, buffer, n_samples, **kwargs):
        new_buffer = defaultdict(list)
        get_predictions = kwargs.get("get_predictions")
        n_neighbors = kwargs.get("n_neighbors", 5)
        device = kwargs.get("device")

        for label, samples in buffer.items():
            if len(samples) <= n_samples:
                new_buffer[label] = samples
                continue

            X = samples.cpu().numpy()

            predictions = get_predictions(samples.to(device))
            pred_labels = predictions.argmax(dim=1).cpu().numpy()

            boundary_scores = self._compute_boundary_scores(
                X, pred_labels, n_neighbors=n_neighbors
            )

            top_indices = np.argsort(boundary_scores)[-n_samples:]

            selected_samples = torch.stack([samples[i] for i in top_indices])
            new_buffer[label] = selected_samples
            # new_buffer[label] = torch.cat([selected_samples, mixup_embeddings(selected_samples)])

        return new_buffer


class CentroidSampling(BaseSampler):
    def sample(self, buffer, n_samples, **kwargs):
        new_buffer = defaultdict(list)

        # Calculate centroid for each class
        for label, samples in buffer.items():
            if len(samples) <= n_samples:
                new_buffer[label] = samples
                continue

            X = samples.cpu().numpy()

            nn = NearestNeighbors(metric="euclidean")
            nn.fit(X)

            centroid = np.mean(X, axis=0)

            indices = nn.kneighbors(
                [centroid], n_neighbors=n_samples, return_distance=False
            )

            new_buffer[label] = samples[indices].squeeze()

        return new_buffer


class KMeansSampling(BaseSampler):
    def sample(self, buffer, n_samples, **kwargs):
        new_buffer = defaultdict(list)

        for label, samples in buffer.items():
            if len(samples) <= n_samples:
                new_buffer[label] = samples
                continue

            X = torch.stack(samples).numpy()

            kmeans = KMeans(n_clusters=n_samples, random_state=0, algorithm="elkan")
            kmeans.fit(X)

            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(X)

            closest_points = []

            for center in kmeans.cluster_centers_:
                _, indices = nn.kneighbors([center])
                closest_points.append(samples[indices[0][0]])

            new_buffer[label] = closest_points

        return new_buffer


class TypicalitySampling(BaseSampler):
    def sample(self, buffer, n_samples, **kwargs):
        batch_size = 200
        x_memory, y_memory = [], []

        new_buffer = {}
        k = 20

        for label, samples in buffer.items():
            if len(samples) > 0:
                samples = torch.stack([x for x, _ in samples])

                kmeans = KMeans(
                    n_clusters=batch_size, random_state=0, algorithm="elkan"
                )
                clusters = kmeans.fit_predict(samples.numpy())

                distances = torch.cdist(samples, samples, p=2.0)

                typicality = torch.zeros(samples.shape[0])

                for i in range(samples.shape[0]):
                    _, nearest_neighbors = torch.topk(
                        distances[i], k=k + 1, largest=False
                    )
                    nearest_neighbors = nearest_neighbors[1:]
                    neighbor_distances = distances[i, nearest_neighbors]

                    typicality[i] = -neighbor_distances.mean()

                for cluster_idx in range(batch_size):
                    cluster_samples = (clusters == cluster_idx).nonzero()[0]
                    if len(cluster_samples) > 0:
                        best_example_idx = cluster_samples[
                            torch.argmax(typicality[cluster_samples])
                        ]
                        if label not in new_buffer:
                            new_buffer[label] = []
                        new_buffer[label].append(samples[best_example_idx])

        for label in new_buffer:
            selected_samples = torch.stack(new_buffer[label])
            print(selected_samples.shape)

            x_memory.extend([selected_sample for selected_sample in selected_samples])
            y_memory.extend([label for _ in range(selected_samples.shape[0])])

        return torch.stack(x_memory), torch.tensor(y_memory, dtype=torch.long)


class HybridSampling(BaseSampler):
    def __init__(self):
        self.boundary_sampler = BoundarySampling()
        self.centroid_sampler = CentroidSampling()

    def sample(self, buffer, n_samples, **kwargs):
        boundary_buffer = self.boundary_sampler.sample(buffer, n_samples // 2, **kwargs)
        centroid_buffer = self.centroid_sampler.sample(buffer, n_samples // 2, **kwargs)

        hybrid_buffer = defaultdict(list)

        for label in boundary_buffer.keys():
            hybrid_buffer[label] = torch.cat(
                [boundary_buffer[label], centroid_buffer[label]]
            )

        assert boundary_buffer.keys() == centroid_buffer.keys() == hybrid_buffer.keys()

        return hybrid_buffer
