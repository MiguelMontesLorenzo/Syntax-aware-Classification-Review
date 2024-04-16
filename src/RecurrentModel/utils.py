import torch
import os
import json
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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

    # save scripted model
    torch.save(model, f"models/{name}.pt")


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function computes the accuracy.

    Args:
    - predictions (torch.Tensor): predictions tensor. Dimensions: [batch, num classes] or [batch].
    - targets (torch.Tensor): targets tensor. Dimensions: [batch, 1] or [batch].

    Returns:
    - accuracy_measure: the accuracy in a tensor of a single element.
    """
    maximums: torch.Tensor = torch.argmax(predictions, dim=1)

    correct_predictions: torch.Tensor = torch.sum(maximums == targets)

    accuracy_measure: torch.Tensor = correct_predictions / len(targets)

    return accuracy_measure


def plot_embeddings(model: torch.nn.Module, int_to_vocab: Dict[int, str], viz_words: int = 400, figsize: Tuple[int, int] = (16, 16)):
    """
    Plots a subset of word embeddings in a 2D space using t-SNE.

    Args:
    - model (torch.nn.Module): The trained SkipGram model containing the embeddings.
    - int_to_vocab (Dict[int, str]): Dictionary mapping word indices back to words.
    - viz_words (int): Number of words to visualize.
    - figsize (Tuple[int, int]): Size of the figure for the plot.
    """
    # Extract embeddings
    embeddings = model.in_embed.weight.to('cpu').data.numpy()
    
    # Reduce the dimensionality of embeddings with t-SNE
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])
    
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
    
    plt.show()


def load_pretrained_weights(pretrained_weights_path, big_vocab_to_int, vocab_to_int) -> torch.Tensor:
    state_dict = torch.load(pretrained_weights_path, map_location=torch.device("cpu"))["in_embed.weight"]

    with open(big_vocab_to_int, "r") as file:
        previous_vocab_to_int = json.loads(file.read())

    indices: List[int] = [previous_vocab_to_int[key] for key in vocab_to_int.keys()]

    embeddings: torch.nn.Embedding = state_dict[indices]
    weights: torch.Tensor = embeddings

    return weights
