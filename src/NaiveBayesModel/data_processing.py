import os
import re
import requests
import zipfile
from collections import Counter

import torch

from src.treebank import Tree
from src.NaiveBayesModel.utils import tokenize


def load_sentences(infile: str) -> tuple[list[str], list[int]]:

    """
    Loads standford dataset sentences (and labels) without the tree structure.

    Args:
        infile (str): 
            Path to the file with the sentences to load.

    Returns:
        (tuple[list[str], list[int]]): 
            A tuple of lists containing the sentences and their corresponding labels.
    """

    sentences: list[str] = list([])
    labels: list[int] = list([])

    with open(infile, "r") as file:
        lines: list[str] = file.readlines()

    for line in lines:
        tree: Tree = Tree(line)
        concat: str = " ".join(tree.get_words())
        sentences.append(tokenize(concat))
        labels.append(tree.labels[-1])

    return (sentences, labels)


def build_vocab(sentences: str) -> tuple[dict[str, int], dict[int, str]]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and
    values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is
        mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    vocabulary: set[str] = set()
    for sentence in sentences:
        vocabulary |= set(sentence)

    wrd2idx: dict[str, int] = {word: i for i, word in enumerate(vocabulary)}
    idx2wrd: dict[int, str] = {i: word for word, i in wrd2idx.items()}

    return (wrd2idx, idx2wrd)


def bag_of_words(text: list[str], vocab: dict[str, int], binary: bool = False) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided
    vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as
        keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW
        representation.

    Returns:
        (torch.Tensor): A tensor representing the bag-of-words vector.
    """
    # Initialize a counter for the text
    word_counter = Counter(text)

    # Create a BoW vector
    bow = torch.zeros(len(vocab), dtype=torch.float32)

    if binary:
        # Update the BoW vector for each word in the text
        for word in word_counter:
            if word in vocab:
                bow[vocab[word]] = 1
    else:
        # Update the BoW vector for each word in the text
        for word, count in word_counter.items():
            if word in vocab:
                bow[vocab[word]] = count

    return bow
