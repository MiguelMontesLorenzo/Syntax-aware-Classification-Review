from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import os
import json


def clean_text(text: str) -> str:
    """
    Replaces symbols with characters that the model can use.

    Args:
    - text (str): text to be processed.

    Returns:
    - text (str): proecessed text.
    """
    substitutions: Dict[str, str] = {
        ".": " <PERIOD> ",
        ",": " <COMMA> ",
        '"': " <QUOTATION_MARK> ",
        ";": " <SEMICOLON> ",
        "!": " <EXCLAMATION_MARK> ",
        "?": " <QUESTION_MARK> ",
        "(": " <LEFT_PAREN> ",
        ")": " <RIGHT_PAREN> ",
        "--": " <HYPHENS> ",
        "?": " <QUESTION_MARK> ",
        ":": " <COLON> ",
    }
    text = text.lower()
    for key, value in substitutions.items():
        text.replace(key, value)
    return text


def tokenize(sentences: List[str]) -> Tuple[List[str], Dict[int, str]]:
    """
    Processed the input sentences to extrcat the list of words and tokenize the text.

    Args:
    - sentences (List[str]): list of correctly processed sentences fomr both files.

    Returns:
    - trimmed_words (List[str]): list of trimmed (filtered) words.
    - correspondences (Dict[int, Tuple(int)]): association of word index in tokens and
    the index of the sentence that
    word belongs to.
    """
    correspondences: Dict[int, str] = {}

    trimmed_words: List[str] = []
    index: int = 0
    for i in range(len(sentences)):
        splitted_sentence: List[str] = sentences[i].split()
        for j in range(len(splitted_sentence)):
            word: str = splitted_sentence[j].lower()
            trimmed_words.append(word)
            correspondences[index] = (i, j)
            index += 1

    return trimmed_words, correspondences


def plot_embeddings(
    model, int_to_vocab: Dict[int, str], viz_words=400, figsize=(16, 16)
) -> None:
    """
    Plots a subset of word embeddings in a 2D space using t-SNE.

    Args:
    - model (SkipGram): trained SkipGram model containing the embeddings.
    - int_to_vocab (Dict[int, str]): Dictionary mapping word indices back to words.
    - viz_words (int): Number of words to visualize.
    - figsize (Tuple[int]): Size of the figure for the plot.

    Returns:
    - None
    """
    # Extract embeddings
    embeddings = model.in_embed.weight.to("cpu").data.numpy()

    # Reduce the dimensionality of embeddings with t-SNE
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color="steelblue")
        plt.annotate(
            int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7
        )

    plt.show()


def save_model(model, model_path="skipgram_dep_model.pth"):
    """
    Save the trained SkipGram model to a file, creating the directory if it
    does not exist.

    Args:
    - model (SkipGram): trained SkipGram model.
    - model_path (str): path to save the model file, including directory
    and filename.

    Returns:
    - model_path (str): path where the model was saved.
    """
    # Extract the directory path from the model_path
    directory = os.path.dirname(model_path)

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), model_path)
    return model_path


def write_dict(json_file: str, vocab_to_int: Dict[str, int]) -> None:
    with open(json_file, "w") as outfile:
        json.dump(vocab_to_int, outfile)
