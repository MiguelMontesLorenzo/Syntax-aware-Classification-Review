from typing import List, Dict

import torch
import numpy as np
import re
import string


def remove_punctuations(input_col):
    """To remove all the punctuations present in the text.Input the text column"""
    table = str.maketrans("", "", string.punctuation)
    return input_col.translate(table)


# Tokenizes a input_string. Takes a input_string (a sentence), splits out punctuation
# and contractions, and returns a list of strings, with each input_string being a token.
def tokenize(input_string: str) -> list[str]:
    input_string = input_string.lower()
    input_string = input_string.strip()

    input_string = remove_punctuations(input_string)
    input_string = re.sub(r"[^A-Za-z0-9(),.!?\'`\-\"]", " ", input_string)
    input_string = re.sub(r"\'s", " 's", input_string)
    input_string = re.sub(r"\'ve", " 've", input_string)
    input_string = re.sub(r"n\'t", " n't", input_string)
    input_string = re.sub(r"\'re", " 're", input_string)
    input_string = re.sub(r"\'d", " 'd", input_string)
    input_string = re.sub(r"\'ll", " 'll", input_string)
    input_string = re.sub(r"\.", " . ", input_string)
    input_string = re.sub(r",", " , ", input_string)
    input_string = re.sub(r"!", " ! ", input_string)
    input_string = re.sub(r"\?", " ? ", input_string)
    input_string = re.sub(r"\(", " ( ", input_string)
    input_string = re.sub(r"\)", " ) ", input_string)
    input_string = re.sub(r"\-", " - ", input_string)
    input_string = re.sub(r"\"", ' " ', input_string)
    # We may have introduced double spaces, so collapse these down
    input_string = re.sub(r"\s{2,}", " ", input_string)
    return list(filter(lambda x: len(x) > 0, input_string.split(" ")))


class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[str]): List of words.
        label (int): Sentiment label (0 for negative, 1 for positive).
    """

    def __init__(self, words: List[str], label: int):
        self._words: list[str] = words
        self._label: int = label

    def __repr__(self) -> str:
        if self.label is not None:
            return f"{self.words}; label={self.label}"
        else:
            return f"{self.words}, no label"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, SentimentExample):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.words == other.words and self.label == other.label

    @property
    def words(self) -> list[str]:
        return self._words

    @words.setter
    def words(self, value):
        raise NotImplementedError

    @property
    def label(self) -> int:
        return self._label

    @label.setter
    def label(self, value):
        raise NotImplementedError


def evaluate_classification(
    predictions: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate classification metrics including accuracy, precision, recall, and F1-score.

    Args:
        predictions (torch.Tensor): Predictions from the model.
        labels (torch.Tensor): Actual ground truth labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """

    accuracy = torch.sum(predictions == labels) / len(labels)
    metrics = {
        "accuracy": accuracy.item(),
    }

    return metrics


def randomize_indices(N: int) -> np.ndarray:
    """
    Randomly shuffles the rows of a 2D tensor.

    """

    indices = np.arange(N)
    shuffled_indices = np.random.permutation(indices)

    return shuffled_indices


def list_random_shuffle(input_list: list[int]) -> list[int]:
    """
    Randomly shuffles the rows of a 2D tensor.

    """
    array: np.ndarray = np.array(input_list)
    random_indices: np.ndarray = randomize_indices(array.shape())
    array: np.ndarray = array[random_indices]
    shuffled_list: list[int] = array.tolist()

    return shuffled_list
