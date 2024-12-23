import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
from TextDataProcessor import TextDataProcessor
from KNNClassifier import KNNClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score


class TextClassifier:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str = 'cpu') -> None:
        """
        Класс для классификации текстов с использованием KNN-классификатора и эмбеддингов.

        Args:
            model_name (str): Название модели для генерации эмбеддингов.
            device (str): Устройство для выполнения операций ('cpu' или 'cuda').
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = TextDataProcessor(tokenizer=self.tokenizer, model=self.model, device=device)
        self.knn = KNNClassifier(device=device)
        self.class_names = []
        self._texts = []
        self._labels = []
        self._idxs = []
        self._new_texts = None
        self._new_labels = None
        self._new_idxs = None

    def fit(self) -> None:
        """
        Обучение классификатора на предоставленных текстах и метках.
        """
        if self._new_texts is not None:
            self.add_to_training_data(self._new_texts, self._new_labels)
            self._texts += self._new_texts
            self._new_texts = None
            self._labels += self._new_labels
            self._new_labels = None
            self._idxs += self._new_idxs
            self._new_idxs = None

        self.processor.save_embeddings(self.knn.features, self._idxs, 'embeddings.csv')

    def add_to_training_data(self, new_texts: List[str], new_labels: List[str]) -> None:
        """
        Добавление новых документов к обучающей выборке.

        Args:
            new_texts (List[str]): Список новых текстов для добавления.
            new_labels (List[str]): Список меток классов для новых текстов.
        """
        new_embeddings, new_encoded_labels, _ = self.processor.preprocess_data(new_texts, new_labels)

        # Обновляем список классов, если появились новые
        for label in new_labels:
            if label not in self.class_names:
                self.class_names.append(label)

        self.knn.n_classes = len(self.class_names) + 1  # Учитываем неизвестный класс

        # Добавляем новые данные к существующим
        if self.knn.features is not None and self.knn.labels is not None:
            self.knn.features = torch.cat((self.knn.features, new_embeddings.to(self.device)), dim=0)
            self.knn.labels = torch.cat((self.knn.labels, new_encoded_labels.to(self.device)), dim=0)
        else:
            self.knn.features = new_embeddings.to(self.device)
            self.knn.labels = new_encoded_labels.to(self.device)

    def load_new_documents(self, data_dir: str) -> None:
        """
        Загрузка новых документов из указанной папки.

        Args:
            data_dir (str): Путь к директории с новыми данными.
        """
        self._new_texts, self._new_idxs, self._new_labels = self.processor.load_data(data_dir)

    def predict(self, texts: List[str]) -> List[str]:
        """
        Предсказание классов для предоставленных текстов.

        Args:
            texts (List[str]): Список текстов для классификации.

        Returns:
            List[str]: Список предсказанных меток классов.
        """
        embeddings = self.processor.texts2embeddings(texts)
        predictions = self.knn.predict(embeddings)
        predicted_labels = [self.class_names[label] if label < len(self.class_names)
                            else 'unknown' for label in predictions]
        return predicted_labels



    def evaluate(self, texts: List[str], labels: List[str]) -> dict:
        """
        Оценка качества классификации на предоставленных данных.

        Args:
            texts (List[str]): Список текстов для оценки.
            labels (List[str]): Список истинных меток классов.

        Returns:
            dict: Метрики оценки (accuracy, f1, precision, recall, balanced_accuracy).
        """
        embeddings = self.processor.texts2embeddings(texts)
        predictions = self.knn.predict(embeddings)

        # Кодирование меток классов
        known_classes = self.processor.label_encoder.classes_
        encoded_labels = []

        for label in labels:
            if label in known_classes:
                encoded_labels.append(self.processor.label_encoder.transform([label])[0])
            else:
                # Присваиваем новый индекс для неизвестных классов
                encoded_labels.append(len(known_classes))

        encoded_labels = torch.tensor(encoded_labels, device=self.device)
        metrics = {
            'accuracy': accuracy_score(encoded_labels, predictions),
            'f1': f1_score(encoded_labels, predictions, average='weighted'),
            'precision': precision_score(encoded_labels, predictions, average='weighted'),
            'recall': recall_score(encoded_labels, predictions, average='weighted'),
            'balanced_accuracy': balanced_accuracy_score(encoded_labels, predictions),
        }
        return metrics

    def save_model(self, file_path: str) -> None:
        """
        Сохранение классификатора в файл.

        Args:
            file_path (str): Путь к файлу для сохранения.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'knn_features': self.knn.features,
            'knn_labels': self.knn.labels,
        }, file_path)

    def load_model(self, file_path: str) -> None:
        """
        Загрузка классификатора из файла.

        Args:
            file_path (str): Путь к файлу для загрузки.
        """
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint['class_names']
        self.knn.features = checkpoint['knn_features']
        self.knn.labels = checkpoint['knn_labels']
