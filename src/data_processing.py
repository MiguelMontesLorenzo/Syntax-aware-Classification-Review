import os
import re
import requests
import zipfile
from collections import Counter

import torch

from src.treebank import Tree
from src.utils import tokenize


def download_file(url: str, save_dir: str, filename: str) -> None:
    """
    Args:
    - url (str): url of the dataset.
    - save_dir (str): directory for the dataset.
    - file_name (str): name of the file.

    Returns:
    - None
    """
    # Get the file path to save to
    filepath = os.path.join(save_dir, filename)

    # Download the file
    print(f"Downloading {filename}...")
    response = requests.get(url)
    with open(filepath, "wb") as file:
        file.write(response.content)

    return filepath


def extract_zip(zip_path: str, extract_dir: str) -> None:
    """
    Function to extract the files of zip_file into extract_dir.

    Args:
    - zip_path (str): Path to the zip file to unzip.
    - extract_dir (str): Path to load the unzipped files.

    Returns:
    - None
    """

    print(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def download_data() -> str:
    """
    Complete pipeline to download sst data.

    Args:
    - None

    Returns:
    - None
    """

    # Define the URL, directory name, and file name
    urls: list[str] = ["http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"]
    path: str = "data_sst/trees"
    name: str = path + "/sst.zip"

    # unzipped files
    trn_file: str = os.path.join(path, "train.txt")  # Path to extracted train file
    val_file: str = os.path.join(path, "dev.txt")  # Path to extracted validation file
    tst_file: str = os.path.join(path, "test.txt")  # Path to extracted test file

    if not os.path.exists(path):
        os.makedirs(path)

        # Download the file
        zip_path = download_file(urls[0], ".", name)

        # Extract the ZIP file
        extract_zip(zip_path, path)

    return (trn_file, val_file, tst_file)


def load_sentences(infile: str) -> tuple[list[str], list[int]]:

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


# def bag_of_words(
#     text: list[str], vocab: dict[str, int], binary: bool = False
# ) -> torch.Tensor:
#     """
#     Converts a list of words into a bag-of-words vector based on the provided
#     vocabulary.
#     Supports both binary and full (frequency-based) bag-of-words representations.

#     Args:
#         text (List[str]): A list of words to be vectorized.
#         vocab (Dict[str, int]): A dictionary representing the vocabulary with words as
#         keys and indices as values.
#         binary (bool): If True, use binary BoW representation; otherwise, use full BoW
#         representation.

#     Returns:
#         (torch.Tensor): A tensor representing the bag-of-words vector.
#     """
#     # TODO: Converts list of words into BoW, take into account the binary vs full

#     bow: torch.Tensor

#     if binary:
#         bow = torch.Tensor([word in text for word in vocab])
#     else:
#         filtered_text: list[str] = [word for word in text if word in vocab]
#         frequency_counter = Counter({element: 0 for element in vocab.keys()})
#         frequency_counter.update(filtered_text)

#         bow: torch.Tensor = torch.zeros(size=[len(vocab)], dtype=torch.float32)
#         for word, i in vocab.items():
#             bow[i] = frequency_counter[word]

#     return bow


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


def save_bows(bows: list[torch.Tensor], path: str) -> None:
    """
    Saves a BoW vector to a file.
    """

    # Saves a file with each line containing the indices of non-zero elements its
    # corresponding sentence BoW vector
    with open(path, "w") as file:
        for bow in bows:
            file.write(" ".join([str(idx) for idx in torch.nonzero(bow)]) + "\n")

    return None

def load_bows(path: str, vocab_size: int) -> list[torch.Tensor]:
    """
    Loads BoW vectors from a file into a list of torch.Tensor objects.

    Args:
        path (str): 
            The path to the file containing the saved BoW vectors.
        vocab_size (int): 
            The size of the vocabulary, which is also the size of the BoW vectors.

    Returns:
        list[torch.Tensor]: A list of BoW vectors loaded from the file.
    """

    bows = []

    with open(path, "r") as file:
        for line in file:
            bow = torch.zeros(vocab_size, dtype=torch.float32)
            indices = list(map(int, line.strip().split()))
            bow[indices] = 1
            bows.append(bow)

    return bows