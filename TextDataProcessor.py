import os
import re
import csv
import nltk
import torch
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel


nltk.download('punkt')
nltk.download('stopwords')


class TextDataProcessor:
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModel, device: str = 'cpu',
                 languages: List[str] = ['english', 'russian']) -> None:
        """
        Класс для загрузки, обработки текстовых данных и их преобразования в эмбеддинги.

        Args:
            tokenizer: Токенизатор для преобразования текста.
            model: Модель для генерации эмбеддингов.
            device (str): Устройство для выполнения операций ('cpu' или 'cuda').
            languages (List[str]): Список языков для загрузки стоп-слов.
        """
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.label_encoder = LabelEncoder()
        self.stop_words = self._load_stopwords(languages)  # Загрузка стоп-слов для указанных языков

    def _load_stopwords(self, languages: List[str]) -> set:
        """
        Загрузка стоп-слов для нескольких языков.

        Args:
            languages (List[str]): Список языков для загрузки стоп-слов.

        Returns:
            set: Множество стоп-слов для указанных языков.
        """
        stop_words = set()
        for language in languages:
            try:
                stop_words.update(stopwords.words(language))
            except OSError:
                print(f"Предупреждение: Стоп-слова для языка '{language}' не найдены.")
        return stop_words

    def save_embeddings(self, embeddings: torch.Tensor, ids: List[int], file_path: str) -> None:
        """
        Сохранение эмбеддингов в файл.

        Args:
            embeddings (torch.Tensor): Тензор эмбеддингов размерности (num_samples, embedding_dim).
            ids (List[int]): Список ID для каждого эмбеддинга.
            file_path (str): Путь к файлу для сохранения данных.
        """
        if embeddings.size(0) != len(ids):
            raise ValueError("Количество эмбеддингов не соответствует количеству ID.")

        # Сохраняем данные в формате CSV
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Записываем заголовок
            writer.writerow(['id'] + [f'dim_{i}' for i in range(embeddings.size(1))])

            # Записываем эмбеддинги и их ID
            for idx, embedding in zip(ids, embeddings):
                writer.writerow([idx] + embedding.tolist())

    def load_embeddings(self, file_path: str) -> Tuple[torch.Tensor, List[str]]:
        """
        Загрузка эмбеддингов из файла.

        Args:
            file_path (str): Путь к файлу с сохраненными эмбеддингами.

        Returns:
            Tuple[torch.Tensor, List[str]]:
                - torch.Tensor: Тензор эмбеддингов размерности (num_samples, embedding_dim).
                - List[str]: Список ID для каждого эмбеддинга.
        """
        embeddings = []
        ids = []

        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # Пропускаем заголовок
            for row in reader:
                ids.append(int(row[0]))  # Первый столбец — ID
                embeddings.append([float(value) for value in row[1:]])  # Остальные столбцы — эмбеддинги

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        return embeddings_tensor, ids

    def load_data(self, data_dir: str) -> Tuple[List[str], List[int], List[str]]:
        """
        Загрузка данных из директории.

        Args:
            data_dir (str): Путь к директории с данными.

        Returns:
            Tuple[List[str], List[int], List[str]]:
                - texts (List[str]): Список текстов.
                - idxs (List[int]): Индексы классов.
                - labels (List[str]): Метки классов.
        """
        texts = []
        labels = []
        idxs = []
        class_names = os.listdir(data_dir)
        class_names = [item for item in class_names if not item.startswith('.')]
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(text)
                    idxs.append(label)
                    labels.append(class_name)
        return texts, idxs, labels

    def preprocess_text(self, text: str) -> str:
        """
        Предобработка текста.

        Args:
            text (str): Исходный текст.

        Returns:
            str: Предобработанный текст.
        """
        # Удаление HTML-тегов
        text = re.sub(r'<.*?>', '', text)

        # Удаление неалфавитных символов и приведение к нижнему регистру
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

        # Токенизация
        words = word_tokenize(text)

        # Удаление стоп-слов
        filtered_words = [word for word in words if word not in self.stop_words]

        # Объединение токенов обратно в строку
        preprocessed_text = ' '.join(filtered_words)
        return preprocessed_text

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Преобразование текста в эмбеддинг.

        Args:
            text (str): Текст для преобразования.

        Returns:
            torch.Tensor: Эмбеддинг текста.
        """
        text = self.preprocess_text(text)
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return torch.tensor(embeddings[0])

    def texts2embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Преобразование списка текстов в тензор эмбеддингов.

        Args:
            texts (List[str]): Список текстов.

        Returns:
            torch.Tensor: Тензор эмбеддингов.
        """
        embeddings = []
        for text in texts:
            preprocessed_text = self.preprocess_text(text)
            embedding = self.embed_text(preprocessed_text)
            embeddings.append(embedding)

        np_embeddings = np.array([e.numpy() for e in embeddings])
        return torch.tensor(np_embeddings)

    def preprocess_data(self, texts: List[str], labels: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Предобработка данных для обучения.

        Args:
            texts (List[str]): Список текстов.
            labels (List[str]): Список меток классов.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[str]]:
                - torch.Tensor: Тензор эмбеддингов текстов.
                - torch.Tensor: Тензор меток классов.
                - List[str]: Список уникальных имен классов.
        """
        # Преобразуем тексты в эмбеддинги
        embeddings = self.texts2embeddings(texts)

        # Кодируем метки классов в числовой формат
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Преобразуем метки и эмбеддинги в тензоры
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)

        # Возвращаем эмбеддинги, метки и список имен классов
        return embeddings_tensor, labels_tensor, self.label_encoder.classes_.tolist()