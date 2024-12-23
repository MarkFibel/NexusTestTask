import torch.nn.functional as F
from typing import Optional
import torch


class KNNClassifier:
    def __init__(self, k: int = 3, threshold: Optional[float] = 0.14, device: str = 'cpu', n_classes: Optional[int]=None) -> None:
        self.n_classes: int = n_classes  # Including the additional class for unknown
        self.k: int = k
        self.threshold: Optional[float] = threshold
        self.device: str = device
        self.features: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None

    def kneighbors(self, features: torch.Tensor, return_distance: bool = True) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = features.to(self.device)

        # Compute cosine distances
        distances = self._compute_distances(features, self.features)

        # Find the k nearest neighbors
        knn = torch.topk(distances, self.k, largest=False, dim=1)
        knn_indices = knn.indices  # Indices of the k nearest neighbors
        knn_distances = knn.values if return_distance else None  # Distances if requested

        return knn_indices, knn_distances

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.n_classes = labels.unique().sum() + 1
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        features = features.to(self.device)

        # Compute cosine distances
        distances = self._compute_distances(features, self.features)

        # Find the k nearest neighbors
        knn_indices = torch.topk(distances, self.k, largest=False, dim=1).indices
        knn_labels = self.labels[knn_indices]

        # Majority vote
        predicted_labels = self._majority_vote(knn_labels)

        # Apply threshold for unknown class detection
        if self.threshold is not None:
            min_distances, _ = torch.min(distances, dim=1)
            predicted_labels[min_distances > self.threshold] = self.n_classes - 1  # Assign to unknown class

        return predicted_labels

    def _compute_distances(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine distances between x1 and x2.

        Args:
            x1 (torch.Tensor): Tensor of shape (num_samples_1, ...).
            x2 (torch.Tensor): Tensor of shape (num_samples_2, ...).

        Returns:
            torch.Tensor: Pairwise cosine distances of shape (num_samples_1, num_samples_2).
        """
        # Normalize the vectors to unit length
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)

        # Compute cosine similarity
        cosine_similarity = torch.matmul(x1_norm, x2_norm.T)

        # Cosine distance is 1 - cosine similarity
        cosine_distance = 1 - cosine_similarity
        return cosine_distance

    def _majority_vote(self, knn_labels: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(knn_labels, num_classes=self.n_classes)
        votes = one_hot.sum(dim=1)
        predicted_labels = votes.argmax(dim=1)
        return predicted_labels

    def incremental_predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Предсказание с учётом добавления новых классов.

        Args:
            features (torch.Tensor): Эмбеддинги новых документов.

        Returns:
            torch.Tensor: Предсказанные метки.
        """
        features = features.to(self.device)

        # Обновляем эмбеддинги для новых документов
        knn_indices, knn_distances = self.kneighbors(features)

        # Получаем метки ближайших соседей
        knn_labels = self.labels[knn_indices]

        # Прогнозируем классы с помощью голосования
        predicted_labels = self._majority_vote(knn_labels)

        # Применяем порог для обнаружения неизвестных классов
        if self.threshold is not None:
            min_distances, _ = torch.min(knn_distances, dim=1)
            predicted_labels[min_distances > self.threshold] = self.n_classes - 1  # Класс "неизвестно"

        return predicted_labels