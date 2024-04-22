from typing import List, Tuple, Dict, Optional
import os
import requests
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

from src.treebank import Tree
from src.RecursiveModel.utils import flatten


def download_file(url: str, save_dir: str, filename: str) -> str:
    """
    Args:
    - url (str): url of the dataset.
    - save_dir (str): directory for the dataset.
    - file_name (str): name of the file.

    Returns:
    - file_path (str)
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
    encoding: str = "utf-8"
    with open(filename, "r", encoding=encoding) as open_file:
        trees: List[Tree] = [Tree(line.strip()) for line in open_file.readlines()]

    return trees


class SSTDataset(Dataset):
    def __init__(self, data: List[Tree], vocab_to_int: Dict[str, int]):
        """
        Initialize the dataset with the tree data and a vocabulary-to-integer mapping.

        Args:
            tokens (List[str]): The list of preprocessed and tokenized words from
            the text data.
            vocab_to_int (Dict[str, int]): A dictionary mapping words to integers.
            context_size (int): The number of words to include in the context.
        """
        self.vocab_to_int: Dict[str, int] = vocab_to_int
        self.data: List[Tuple[List[Optional[str]], Optional[int]]] = [
            (tree.get_words(), tree.labels[-1] if tree.labels else 0) for tree in data
        ]

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Return a single item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            A tuple of context and target, both converted to integer representations.
        """
        sentence, target = self.data[idx]
        sentence_tensor = torch.tensor(
            [
                self.vocab_to_int[token]
                for token in sentence
                if token in self.vocab_to_int
            ]
        )
        target_tensor = torch.tensor(target)
        return sentence_tensor, target_tensor.squeeze()


def preprocess_data() -> (
    Tuple[List[Tree], List[Tree], List[Tree], Dict[str, int], Dict[int, str]]
):
    # Load training, validation and test data
    train_data: List[Tree] = load_trees("train")
    val_data: List[Tree] = load_trees("dev")
    test_data: List[Tree] = load_trees("test")

    # Create list with all the Trees
    whole_data: List[Tree] = []
    whole_data.extend(train_data)
    whole_data.extend(val_data)
    whole_data.extend(test_data)

    vocab_to_int: Dict[str, int]
    int_to_vocab: Dict[int, str]
    vocab_to_int, int_to_vocab = create_lookup_tables(whole_data)

    return train_data, val_data, test_data, vocab_to_int, int_to_vocab


def generate_dataloaders(
    batch_size=128, num_workers=4
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    train_data, val_data, test_data, vocab_to_int, int_to_vocab = preprocess_data()

    train_dataset: Dataset = SSTDataset(train_data, vocab_to_int)
    val_dataset = SSTDataset(val_data, vocab_to_int)
    test_dataset = SSTDataset(test_data, vocab_to_int)

    # Define dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return train_dataloader, val_dataloader, test_dataloader, vocab_to_int, int_to_vocab


def create_lookup_tables(data: List[Tree]) -> Tuple[Dict[str, int], Dict[int, str]]:
    words: List[str] = [
        word for tree in data for word in tree.get_words() if word is not None
    ]

    word_counts: Counter = Counter(words)
    sorted_vocab: List[str] = sorted(
        word_counts.keys(), key=lambda word: word_counts[word], reverse=True
    )

    int_to_vocab: Dict[int, str] = {i: word for i, word in enumerate(sorted_vocab)}
    vocab_to_int: Dict[str, int] = {word: i for i, word in enumerate(sorted_vocab)}
    return vocab_to_int, int_to_vocab


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares and returns a batch for training/testing in a torch model.

    This function sorts the batch by the length of the text sequences in
    descending order,
    tokenizes the text using a pre-defined word-to-index mapping, pads the
    sequences to have
    uniform length, and converts labels to tensor.

    Args:
        batch (List[Tuple[List[str], int]]): A list of tuples, where each
        tuple contains a list of words (representing a text) and an integer
        label.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
        three elements:
            - texts_padded (torch.Tensor): A tensor of padded word indices of
            the text.
            - labels (torch.Tensor): A tensor of labels.
            - lengths (torch.Tensor): A tensor representing the lengths of
            each text sequence.
    """

    sorted_batch: List[Tuple[torch.Tensor, torch.Tensor]] = list(
        reversed(sorted(batch, key=lambda x: x[0].nelement()))
    )

    texts_indx: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []

    for tensored_text, label in sorted_batch:
        if tensored_text.nelement() > 0:
            texts_indx.append(tensored_text)
            labels.append(label)

    lengths: List[int] = [tensor_text.nelement() for tensor_text in texts_indx]

    lengths_torch: torch.Tensor = torch.tensor(lengths)

    texts_padded: torch.Tensor = pad_sequence(texts_indx, batch_first=True)

    labels_torch: torch.Tensor = torch.tensor(labels)

    return texts_padded, labels_torch, lengths_torch


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
