from typing import List
import torch
import os
import json


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
    - model (torch.nn.Module): pytorch model.
    - name (str): name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    if not os.path.isdir(f"models/{name}"):
        os.makedirs(f"models/{name}")

    # save scripted model
    torch.save(model, f"models/{name}/{name}.pt")

    return None


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function computes the accuracy.

    Args:
    - predictions (torch.Tensor):
        predictions tensor. Dimensions: [batch, num classes] or [batch].
    - targets (torch.Tensor):
        targets tensor. Dimensions: [batch, 1] or [batch].

    Returns:
    - accuracy_measure: the accuracy in a tensor of a single element.
    """
    maximums: torch.Tensor = torch.argmax(predictions, dim=1)
    correct_predictions: torch.Tensor = torch.sum(torch.isclose(maximums, targets))
    accuracy_measure: torch.Tensor = correct_predictions / len(targets)

    return accuracy_measure


def load_pretrained_weights(
    pretrained_weights_path, big_vocab_to_int, vocab_to_int
) -> torch.Tensor:
    state_dict = torch.load(pretrained_weights_path, map_location=torch.device("cpu"))[
        "in_embed.weight"
    ]

    with open(big_vocab_to_int, "r") as file:
        previous_vocab_to_int = json.loads(file.read())

    indices: List[int] = [previous_vocab_to_int[key] for key in vocab_to_int.keys()]

    embeddings: torch.nn.Embedding = state_dict[indices]
    weights: torch.Tensor = embeddings

    return weights
