from typing import List
import torch


def read_indices(filepath: str) -> List[int]:
    """
    Get the indices of the phrases that have negationos.
    Args:
    - filepath(str): path of the file with the desired indices.
    Returns:
    - indices (List[int]): indices of the desired sentences
    """

    with open(filepath, "r") as file:
        text: str = file.readline()

    indices_str: List[str] = text.split(",")

    indices = [int(index.strip()) - 1 for index in indices_str]

    return indices


def count_correct(predictions: torch.Tensor, labels: List[int]) -> int:
    """
    Count correct predictions given the labels.
    Args:
    - predictions (torch.Tensor): list of model predictions (as tensor)
    - labels (List[int]): list of correct labels
    Returns:
    - correct (int): number of correct predictions
    """
    correct: int = 0

    for prediction, label in zip(predictions, labels):
        if prediction.item() == label:
            correct += 1

    return correct
