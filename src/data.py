import torch
from src.treebank import Tree

import os
import requests
import zipfile
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence


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
    urls: list[str] = ["http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"]
    path: str = "data_sst"
    name: str = path + "/sst.zip"

    if not os.path.exists(path):
        os.makedirs(path)

        # Download the file
        zip_path: str = download_file(urls[0], ".", name)

        # Extract the ZIP file
        extract_zip(zip_path, path)


def load_trees(file: str) -> list[Tree]:
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
        trees: list[Tree] = [Tree(line) for line in file.readlines()]

    return trees


class SSTDataset(Dataset):
    def __init__(self, data: list[Tree], vocab_to_int: dict[str, int]) -> None:
        """
        Initialize the dataset with the tree data and a vocabulary-to-integer mapping.

        Args:
            tokens (list[str]):
                The list of preprocessed and tokenized words from the text data.
            vocab_to_int (dict[str, int]):
                A dictionary mapping words to integers.
            context_size (int):
                The number of words to include in the context.
        """
        self.vocab_to_int: dict[str, int] = vocab_to_int
        self.data: list[tuple[list[str], int]] = [
            (tree.get_words(), tree.labels[-1]) for tree in data
        ]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor]:
        """
        Return a single item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            A tuple of context and target, both converted to integer representations.
        """
        sentence, target = self.data[idx]
        sentence_tensor: torch.Tensor = torch.tensor(
            [
                self.vocab_to_int[token]
                for token in sentence
                if token in self.vocab_to_int
            ]
        )
        target_tensor: torch.Tensor = torch.tensor(target)
        return sentence_tensor, target_tensor.squeeze()


def preprocess_data() -> (
    tuple[list[Tree], list[Tree], list[Tree], dict[str, int], dict[int, str]]
):
    # Load training, validation and test data
    train_data: list[Tree] = load_trees("train")
    val_data: list[Tree] = load_trees("dev")
    test_data: list[Tree] = load_trees("test")

    # Create list with all the Trees
    whole_data: list[Tree] = []
    whole_data.extend(train_data)
    whole_data.extend(val_data)
    whole_data.extend(test_data)

    vocab_to_int: dict[str, int]
    int_to_vocab: dict[int, str]
    vocab_to_int, int_to_vocab = create_lookup_tables(whole_data)

    return train_data, val_data, test_data, vocab_to_int, int_to_vocab


def generate_dataloaders(
    batch_size=128, num_workers=4
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:

    train_data: list[Tree]
    val_data: list[Tree]
    test_data: list[Tree]
    vocab_to_int: dict[str, int]
    int_to_vocab: dict[int, str]
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

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        vocab_to_int,
        int_to_vocab,
        train_data,
        val_data,
        test_data,
    )


def create_lookup_tables(data: list[Tree]) -> tuple[dict[str, int], dict[int, str]]:
    words: list[str] = [word for tree in data for word in tree.get_words()]

    word_counts: Counter = Counter(words)
    sorted_vocab: list[int] = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_vocab: dict[int, str] = {i: word for i, word in enumerate(sorted_vocab)}
    vocab_to_int: dict[str, int] = {word: i for i, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares and returns a batch for training/testing in a torch model.

    This function sorts the batch by the length of the text sequences in descending
    order, tokenizes the text using a pre-defined word-to-index mapping, pads the
    sequences to have uniform length, and converts labels to tensor.

    Args:
        batch (list[tuple[list[str], int]]):
            A list of tuples, where each tuple contains a list of words (representing a
            text) and an integer label.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing three elements:
            - texts_padded (torch.Tensor): A tensor of padded word indices of the text.
            - labels (torch.Tensor): A tensor of labels.
            - lengths (torch.Tensor): A tensor representing the lengths of each text
              sequence.
    """

    sorted_batch: list[tuple[list[torch.Tensor, torch.Tensor]]] = reversed(
        sorted(batch, key=lambda x: x[0].nelement())
    )

    texts_indx: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for tensored_text, label in sorted_batch:
        if tensored_text.nelement() > 0:
            texts_indx.append(tensored_text)
            labels.append(label)

    lengths: list[int] = [tensor_text.nelement() for tensor_text in texts_indx]

    lengths: torch.Tensor = torch.tensor(lengths)

    texts_padded: torch.Tensor = pad_sequence(texts_indx, batch_first=True)

    labels: torch.Tensor = torch.tensor(labels)

    return texts_padded, labels, lengths


def get_parsed_sentences(
    trees: list[Tree], word2idx: dict[StopIteration, int]
) -> tuple[list[torch.Tensor], list[int]]:

    sentences: list[torch.Tensor] = []
    labels: list[int] = []

    for tree in trees:
        sentences.append([word2idx(word) for word in tree.get_words()])
        labels.append(tree.labels[-1])

    return (sentences, labels)
