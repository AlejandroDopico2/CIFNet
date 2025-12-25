from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F
from loguru import logger


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
    def sample(self, embeddings: torch.Tensor, n_samples: int) -> torch.Tensor:
        if embeddings.size(0) <= n_samples:
            return embeddings

        z = torch.nn.functional.normalize(embeddings, dim=1)
        mu = z.mean(dim=0)
        dists = torch.norm(z - mu.unsqueeze(0), dim=1)
        indices = torch.argsort(dists)[:n_samples]
        return embeddings[indices]


class KMeansSampling(BaseSampler):
    def sample(self, embeddings: torch.Tensor, n_samples: int) -> torch.Tensor:
        if embeddings.size(0) <= n_samples:
            return embeddings

        z = torch.nn.functional.normalize(embeddings, dim=1).cpu().numpy()
        kmeans = KMeans(n_clusters=n_samples, n_init=10)
        labels = kmeans.fit_predict(z)

        selected = []
        for k in range(n_samples):
            cluster_idx = (labels == k)
            cluster_embs = embeddings[cluster_idx]
            center = torch.tensor(kmeans.cluster_centers_[k])
            dists = torch.norm(
                torch.nn.functional.normalize(cluster_embs, dim=1)
                - center.unsqueeze(0),
                dim=1,
            )
            selected.append(cluster_embs[dists.argmin()])

        return torch.stack(selected)


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

class HerdingSampling(BaseSampler):
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    @torch.no_grad()
    def sample(self, embeddings: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        embeddings: Tensor [N, D] with a single class
        n_samples: number of samples to select
        """
        N, D = embeddings.size()

        if N <= n_samples:
            return embeddings

        z = embeddings
        if self.normalize:
            z = torch.nn.functional.normalize(z, p=2, dim=1)

        mu = z.mean(dim=0)

        selected_indices = []
        selected_sum = torch.zeros_like(mu)

        for k in range(1, n_samples + 1):
            candidate_means = (selected_sum.unsqueeze(0) + z) / k

            dists = torch.norm(candidate_means - mu.unsqueeze(0), dim=1)

            if selected_indices:
                dists[selected_indices] = float("inf")

            idx = torch.argmin(dists).item()
            selected_indices.append(idx)
            selected_sum += z[idx]

        return embeddings[selected_indices]

class PrototypeSampling(BaseSampler):
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    @torch.no_grad()
    def sample(self, embeddings: torch.Tensor, n_samples: int = None) -> torch.Tensor:
        """
        embeddings: Tensor [N, D] (single class)
        returns: Tensor [K, D] prototypes
        """
        N, D = embeddings.size()

        if N <= n_samples:
            return embeddings

        z = embeddings
        if self.normalize:
            z = torch.nn.functional.normalize(z, dim=1)

        z_np = z.cpu().numpy()

        kmeans = KMeans(
            n_clusters=n_samples,
            n_init=10,
            random_state=0,
        )
        kmeans.fit(z_np)
        centers = torch.from_numpy(kmeans.cluster_centers_).to(embeddings.device)

        return centers

class GaussianPrototypeSampling(BaseSampler):
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def sample(self, embeddings: torch.Tensor, n_samples: int = None):
        if embeddings.size(0) <= n_samples:
            return embeddings

        z = embeddings
        if self.normalize:
            z = torch.nn.functional.normalize(z, dim=1)

        mu = embeddings.mean(dim=0)
        sigma = embeddings.std(dim=0)

        prototypes = mu.unsqueeze(0) + torch.randn(
            n_samples, embeddings.size(1), device=embeddings.device
        ) * sigma.unsqueeze(0)

        return prototypes
