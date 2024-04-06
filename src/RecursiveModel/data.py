import os
import requests
import zipfile
from typing import List, Dict, Tuple
from src.RecursiveModel.treebank import Tree
from src.RecursiveModel.utils import flatten


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


def download_data() -> None:
    """
    Complete pipeline to download sst data.

    Args:
    - None

    Returns:
    - None
    """

    # Define the URL, directory name, and file name
    urls: List[str] = ["http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"]
    path: str = "data_sst"
    name: str = path + "/sst.zip"

    if not os.path.exists(path):
        os.makedirs(path)

        # Download the file
        zip_path = download_file(urls[0], ".", name)

        # Extract the ZIP file
        extract_zip(zip_path, path)


def load_trees(file: str) -> List[Tree]:
    """
    Loads training trees. Maps leaf node words to word ids.

    Args:
    - file (str)

    Returns:
    - None
    """
    filename: str = "data_sst/trees/" + file + ".txt"
    print(f"Loading {filename} trees...")
    with open(filename, "r", encoding="utf-8") as file:
        trees: List[Tree] = [Tree(line) for line in file.readlines()]

    return trees


def load_vocab(data: List[Tree]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create the word2index and index2word dictionary from the trees.

    Args:
    - data (List[Tree]): list of trees with the data.

    Returns:
    word2index (Dict[str, int]): Convert word to a unique index
    index2word (Dict[int, str]): Convert form index to string
    """
    vocab: List = list(set(flatten([t.get_words() for t in data])))
    word2index: Dict[str, int] = {"<UNK>": 0}
    for word in vocab:
        if word not in word2index.keys():
            word2index[word] = len(word2index)

    index2word: Dict[int, str] = {v: k for k, v in word2index.items()}

    return word2index, index2word
