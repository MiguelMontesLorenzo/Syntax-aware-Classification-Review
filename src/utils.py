import random
from typing import List
import torch
from src.treebank import Tree
import os


def get_batch(batch_size: int, data: List[Tree]):
    """
    Get batch from the data.
    Args:
    - batch_size (int): batch size.
    - data (List[Tree]): data from which batches are obtained.
    Yields:
    - batch (List[Tree]): batch.
    """
    random.shuffle(data)
    sindex = 0
    eindex = batch_size
    while eindex < len(data):
        batch = data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(data):
        batch = data[sindex:]
        yield batch


def flatten(list: List) -> List:
    """
    Args:
    - list (List)

    Returns:
    - List
    """
    return [item for sublist in list for item in sublist]


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    torch.save(model, f"models/{name}.pt")


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function computes the accuracy.

    Args:
        predictions: predictions tensor. Dimensions:
            [batch, num classes] or [batch].
        targets: targets tensor. Dimensions: [batch, 1] or [batch].

    Returns:
        the accuracy in a tensor of a single element.
    """
    maximums = torch.argmax(predictions, dim=1)

    correct_predictions = torch.sum(maximums == targets)

    accuracy_measure = correct_predictions / len(targets)

    return accuracy_measure
