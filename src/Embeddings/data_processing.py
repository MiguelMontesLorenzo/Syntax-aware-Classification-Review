from typing import List, Tuple, Dict, Set
from collections import Counter
import torch
import random
import spacy
import re
import csv
import subprocess
import os
import requests
import zipfile

from math import sqrt
from torch.utils.data import Dataset, DataLoader

from src.Embeddings.utils import tokenize
from src.treebank import Tree


def load_and_preprocess_data(csv_file: str) -> Tuple[List[str], List[str], Dict[int, Tuple[int, int]], Dict[str, int], Dict[int, str]]:
    """
    Load text and csv data and preprocess it using a tokenize function.

    Args:
    - csv_infile (str): Path to the input csv file containing data form IMBD movie reviews.

    Returns:
    - sentences (List[str]): list of correctly processed sentences fomr both files.
    - tokens (List[str]): list of preprocessed and tokenized words from the input data.
    - correspondences (Dict[int, Tuple[int, int]]): association of word index in tokens and the index of the sentence that
    word belongs to.
    - vocab_to_int (Dict[str, int]): Dictionary mapping words to unique integers.
    - int_to_vocab (Dict[int, str]): Dictionary mapping unique integers to words.
    """
    download_data()

    train_data, val_data, test_data, vocab_to_int, int_to_vocab = preprocess_data()

    sentences: List[str] = process_csv(csv_file)

    vocab_to_int, int_to_vocab = update_lookup_tables(vocab_to_int, int_to_vocab, sentences)
    tree_sentences: List[str] = obtain_sentences(train_data, val_data, test_data)
    sentences.extend(tree_sentences)

    tokens, correspondences = tokenize(sentences)

    return sentences, tokens, correspondences, vocab_to_int, int_to_vocab


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


def preprocess_data() -> Tuple[List[Tree], List[Tree], List[Tree], Dict[str, int], Dict[int, str]]:
    """
    Loads data from Stanford Dataset trees and creates lookup tables for mapping
    integers to vocab and viceversa.

    Args:
    - None

    Returns:
    - train_data (List[Tree]): training examples of Stanford trees.
    - val_data (List[Tree]): validation examples of Stanford trees.
    - test_data (List[Tree]): testing examples of Stanford trees.
    - vocab_to_int (Dict[str, int]): Dictionary mapping words to unique integers.
    - int_to_vocab (Dict[int, str]): Dictionary mapping unique integers to words.
    """
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


def obtain_sentences(train_data, val_data, test_data) -> List[str]:
    """
    Extracts vocabulary sentences from Stanford trees.
    
    Args:
    - train_data (List[Tree]): training examples of Stanford trees.
    - val_data (List[Tree]): validation examples of Stanford trees.
    - test_data (List[Tree]): testing examples of Stanford trees.

    Returns:
    - sentences (List[str]): sentences from data.
    """
    sentences: List[str] = []

    # Create list with all the vocabulary words
    whole_data: List[Tree] = []
    whole_data.extend(train_data)
    whole_data.extend(val_data)
    whole_data.extend(test_data)

    for tree in whole_data:
        # New sentence
        sentence: str = " ".join(tree.get_words())
        splitted_sentences: List[str] = split_sentence(sentence)
        filtered_sentences: List[str] = filter_sentence(splitted_sentences)

        if len(filtered_sentences) > 0:
            sentences.extend(filtered_sentences)

    return sentences


def create_lookup_tables(data: List[Tree]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates lookup tables that match vocabulary words to unique indexes and their
    correpondence the other way around.

    Args:
    - data (List[Tree]): all the trees from the Stanford Dataset.

    Returns:
    - vocab_to_int (Dict[str, int]): Dictionary mapping words to unique integers.
    - int_to_vocab (Dict[int, str]): Dictionary mapping unique integers to words.
    """
    words: List[str] = [word for tree in data for word in tree.get_words()]

    word_counts: Counter = Counter(words)
    sorted_vocab: List[int] = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_vocab: Dict[int, str] = {i: word for i, word in enumerate(sorted_vocab)}
    vocab_to_int: Dict[str, int] = {word: i for i, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab


def update_lookup_tables(vocab_to_int: Dict[str, int], int_to_vocab: Dict[int, str], sentences: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Updates previously built dictionaries with the vocabulary from the IMBD movies reviews.

    Args:
    - vocab_to_int (Dict[str, int]): Dictionary mapping words to unique integers.
    - int_to_vocab (Dict[int, str]): Dictionary mapping unique integers to words.
    - sentences (List[str]): sentences from the IMBD reviews.

    Returns:
    - vocab_to_int (Dict[str, int]): updated dictionary mapping words to unique integers.
    - int_to_vocab (Dict[int, str]): updated dictionary mapping unique integers to words.
    """
    new_words: List[str] = [word for sentence in sentences for word in sentence.split() if word not in vocab_to_int]
    
    word_counts: Counter = Counter(new_words)
    sorted_vocab: List[int] = sorted(word_counts, key=word_counts.get, reverse=True)

    n_words: int = len(vocab_to_int)
    for i, word in enumerate(sorted_vocab):
        index: int = n_words + i
        vocab_to_int[word] = index
        int_to_vocab[index] = word
    
    return vocab_to_int, int_to_vocab


def process_csv(csv_file: str) -> List[str]:
    """
    Process the csv infile with the reviews from the IMBD movies reviews to obtain
    the processed sentences.

    Args:
    - csv_file (str): path to the csv file.

    Returns:
    - sentences (List[str]): sentences from the IMBD reviews.
    """
    encoding: str = "utf-8"
    sentences: list = []
    with open(csv_file, "r", newline="", encoding=encoding) as csv_file:
        csv_reader = csv.reader(csv_file)

        # The second element is the score, irrelevant for the task
        first_sentences: List[str] = [review[0] for review in csv_reader]

        for elem in first_sentences:
            cleaned_sentence = clean_csv(elem)
            splitted_sentences: List[str] = split_sentence(cleaned_sentence)
            filtered_sentences: List[str] = filter_sentence(splitted_sentences)

            if len(filtered_sentences) > 0:
                sentences.extend(filtered_sentences)
        return sentences


def clean_csv(sentence: str) -> str:
    """
    Replaces incorrectly formatted characters.
    
    Args:
    - sentence (str): sentence to be cleaned.

    Returns:
    - sentence (str): cleaned sentence.
    """
    substitutions: Dict[str, str] = {
        "`` ": "",
        "''": "",
        "` ": "",
        "' ": " ",
        " '": " ",
        "<br /><br />": " ",
        "\'s": "'s",
        "/": " ",
        "(": " ",
        ")": " ",
        '"': '',
        "*": "",
        "-": " ",
        "<": "",
        ">": ""
    }

    for key, value in substitutions.items():
        sentence = sentence.replace(key, value)
    return sentence.lower()


def split_sentence(sentence: str) -> List[str]:
    """
    Splits compound sentences into simpler ones where dependency parsing is possible.

    Args:
    - sentence (str): sentence to be splitted.

    Returns:
    - new_sentences (List[str]): sentences splitted from the original one.
    """
    splitters: List[str] = [" and ", ".", ",", ";", "--", ":", "!", "?"]
    previous_sentence_list: List[str] = [sentence]

    for splitter in splitters:
        new_sentences: List[str] = []
        for sentence in previous_sentence_list:
            new_sentences.extend([splitted.strip() for splitted in sentence.split(splitter)])
        previous_sentence_list = new_sentences

    return new_sentences


def filter_sentence(sentences: List[str]) -> List[str]:
    """
    Filters non-existent and short sentences.

    Args:
    - sentences (List[str]): sentences to be filtered.

    Returns:
    - filtered_sentences (List[str]): filtered sentences.
    """
    filtered_sentences: List[str] = [sentence for sentence in sentences if sentence is not None and len(sentence.split()) >= 3]
    return filtered_sentences



def subsample_words(words: List[str], vocab_to_int: Dict[str, int], correspondences: Dict[int, str], threshold: float = 6e-1) -> Tuple[List[int], Dict[str, float], Dict[int, str]]:
    """
    Perform subsampling on a list of word integers using PyTorch, aiming to reduce the 
    presence of frequent words according to Mikolov's subsampling technique. This method 
    calculates the probability of keeping each word in the dataset based on its frequency, 
    with more frequent words having a higher chance of being discarded. The process helps 
    in balancing the word distribution, potentially leading to faster training and better 
    representations by focusing more on less frequent words. It also updates the correspondences
    dictionary to only have the indexes of the sampled words.
    
    Args:
    - words (List[str]): List of words to be subsampled.
    - vocab_to_int (Dict[str, int]): Dictionary mapping words to unique integers.
    - correspondences (Dict[int, str]): association of word index in tokens and the index of the sentence that
    word belongs to.
    - threshold (float): Threshold parameter controlling the extent of subsampling.
  
    Returns:
    - train_words (List[int]): a list of integers representing the subsampled words, where some high-frequency words may be removed.
    - freqs (Dict[str, float]): associates each word with its frequency.
    - sampled_correspondences (Dict[int, Tuple(int)]):: association of word index in tokens and the index of the sentence that
    word belongs to.
    """
    sampled_correspondences: Dict[int, str] = {}
    aux_correspondences: Dict[int, str] = {}
    int_words: List[int] = []
    new_words: List[str] = []
    aux_index: int = 0
    for i, word in enumerate(words):
        if word in vocab_to_int:
            new_words.append(word)
            int_words.append(vocab_to_int[word])
            aux_correspondences[aux_index] = correspondences[i]
            aux_index += 1

    n_words: int = len(new_words)
    freqs: Dict[str, float] = {word: freq/n_words for (word, freq) in Counter(new_words).items()}
    train_words: List[int] = []

    index: int = 0
    for i, word in enumerate(new_words):
        if random.random() < sqrt(threshold/freqs[word] * 2):
            train_words.append(int_words[i])
            sampled_correspondences[index] = aux_correspondences[i]
            index += 1

    return train_words, freqs, sampled_correspondences


def get_neighbours(tree: spacy.tokens.doc.Doc, idx: int, print_idx) -> List[str]:
    """
    Obtains the neighbour words from the dependency tree. A word is considered neigbour
    if it has a direct or inverse relationship with the traget word.

    Args:
    - tree (spacy.tokens.doc.Doc): dependency tree.
    - idx (int): index of the target word in the phrase.

    Returns:
    - neighbours (List[str]): neighbours of the target word.
    """
    if idx < 0 or idx >= len(tree):
        print(idx)
        return []
    
    target: spacy.tokens.token.Token = tree[idx]
    # print("Target", print_idx, target)
    neighbours: Set[str] = set()

    for token in tree:
        if token != target and (token.head == target or target.head == token) and token.dep_ not in ["det", "prep", "cc"]:
            neighbours.add(token.text.lower())
            try:
                if token.head.head == target:
                    neighbours.add(token.text.lower())
            except:
                pass
            try:
                if target.head.head == token:
                    neighbours.add(token.text.lower())
            except:
                pass
    # print("Neighbours", print_idx, neighbours)
    return list(neighbours)


def get_target(words: List[int], idx: int, dependency_tree: spacy.tokens.doc.Doc, word_idx: int, vocab_to_int: Dict[str, int], window_size: int, context_length: int, int_to_vocab) -> List[str]:
    """
    Gets related words with the target word. Relationships include nearby words and dependent words.

    Args:
    - words (List[int]): the list of words from which context words will be selected.
    - idx (int): the index of the target word.
    - dependency_tree (spacy.tokens.doc.Doc): dependency tree.
    - word_idx (int): index of the target word in the phrase.
    - vocab_to_int (Dict[str, int]): Dictionary mapping words to unique integers.
    - window_size (int): the maximum window size for context words selection.

    Returns:
    - target_words (List[str]): list of words selected as traget words.
    """
    frecuent_words: Set[str] = {"the", "to", "of", "a", "if"}
    words_to_remove: Set[str] = {vocab_to_int[word] for word in frecuent_words}
    probability: float = 0.85
    target_words: Set[str] = set()

    neighbours: List[str] = get_neighbours(dependency_tree, word_idx, idx)
    for neighbour in neighbours:
        neighbour: str = re.sub(r"[^a-zA-Z]", "", neighbour)
        if neighbour in vocab_to_int:
            target_words.add(vocab_to_int[neighbour])

    n_words: int = len(words)
    window_size: int = random.randint(1, window_size)
    target_words_print = []

    for i in range(1, window_size + 1):
        if idx - i >= 0:
            target_words.add(words[idx - i])
            target_words_print.append(int_to_vocab[words[idx - i]])
        if idx + i < n_words:
            target_words.add(words[idx + i])
            target_words_print.append(int_to_vocab[words[idx + i]])
    
    target_words: List[int] = list(target_words)
    filtered_words: List[int]  = []
    
    for word in target_words:
        if word in words_to_remove:
            if random.random() > probability:
                filtered_words.append(word)
        else:
            filtered_words.append(word)

    # print("Nearby words", idx, target_words_print)
    
    while len(filtered_words) < context_length:
        filtered_words.append(vocab_to_int["padding"])
    
    filtered_words = filtered_words[:context_length]

    return filtered_words


class EmbeddingsDataset(Dataset):
    def __init__(self, tokens: List[str], sentences: List[str], correspondences: Dict[int, str], nlp: spacy.language.Language, vocab_to_int: Dict[str, int], int_to_vocab: Dict[int, str], window_size: int = 6) -> None:
        """
        Initialize the dataset with the text data, a vocabulary-to-integer mapping, and the context size.

        Args:
        - tokens (List[int]): the list of preprocessed and tokenized words from the text data.
            vocab_to_int (Dict[str, int]): A dictionary mapping words to integers.
            context_size (int): The number of words to include in the context.
        """
        self.words: List[int] = tokens
        self.sentences: List[str] = sentences
        self.correspondences: Dict[int, str] = correspondences
        self.vocab_to_int: Dict[str, int] = vocab_to_int
        self.int_to_vocab: Dict[int, str] = int_to_vocab
        self.window_size: int = window_size
        self.nlp: spacy.language.Language = nlp
        self.trees: Dict[int, spacy.tokens.doc.Doc] = {}
        self.context_length: int = 5

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.words)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a single item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            A tuple of context and target, both converted to integer representations.
        """
        word: int = self.words[idx]
        # print(f"Word: {idx}", self.int_to_vocab[word])
        
        tree_idx, word_idx = self.correspondences[idx]
        # print(f"Sentence: {idx}", self.sentences[tree_idx])

        if tree_idx not in self.trees:
            dependency_tree: spacy.tokens.doc.Doc = self.nlp(self.sentences[tree_idx])
            self.trees[tree_idx] = dependency_tree
        else:   
            dependency_tree = self.trees[tree_idx]
                
        new_targets: List[int] = get_target(self.words, idx, dependency_tree, word_idx, self.vocab_to_int, self.window_size, self.context_length, self.int_to_vocab)
        input_tensor: torch.Tensor = torch.tensor([word] * len(new_targets))

        # targets = [self.int_to_vocab[target] for target in new_targets]
        # print("Targets:", idx, targets)
        # print()

        targets_tensor: torch.Tensor = torch.tensor(new_targets)
        
        return input_tensor, targets_tensor


def generate_data_loader(tokens: List[int], sentences: List[str], correspondences: Dict[int, str], batch_size: int, vocab_to_int: Dict[str, int], int_to_vocab: Dict[int, str]):
    """
    Load data, preprocess, create lookup tables, generate datasets, and create data loaders.

    Args:
        infile (str): Path to the input file containing text data.
        context_size (int): The size of the context window for the CBOW dataset.
        batch_size (int): Batch size for the data loaders.
        start_token (str):  A character used as the start token for each word.
        end_token (str): A character used as the end token for each word.
        train_pct (float): Percentage of training samples from dataset. 

    Returns:
        tuple: A tuple containing the training and testing DataLoader instances, vocabulary size, and the lookup tables.
    """
    # Generate Dataset
    print(f"Creating dataset...")
    nlp: spacy.language.Language = get_dependency_model()
    embeddings_dataset = EmbeddingsDataset(tokens, sentences, correspondences, nlp, vocab_to_int, int_to_vocab, window_size=6)
    print("Dataset created.")

    print(f"Creating dataloaders with batch size = {batch_size}...")
    dataloader = DataLoader(embeddings_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=True, num_workers=4)
    print("Dataloader created.")

    return dataloader


def get_dependency_model() -> spacy.language.Language:
    model_name: str = "en_core_web_sm"
    try:
        nlp: spacy.language.Language = spacy.load(model_name)
    except:
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        nlp: spacy.language.Language = spacy.load(model_name)
    return nlp



def cosine_similarity(embedding: torch.nn.Embedding, valid_size: int = 16, valid_window: int = 100, device: str = 'cpu'):
    """Calculates the cosine similarity of validation words with words in the embedding matrix.

    This function calculates the cosine similarity between some random words and
    embedding vectors. Through the similarities, it identifies words that are
    close to the randomly selected words.

    Args:
    - embedding (torch.nn.Embedding): a PyTorch Embedding module.
    - valid_size (int): number of random words to evaluate.
    - valid_window (int): the range of word indices to consider for the random selection.
    - device (str): the device (CPU or GPU) where the tensors will be allocated.

    Returns:
    - valid_indices (torch.Tensor): valid examples
    - similarities (torch.Tensor): their cosine similarities with
    the embedding vectors.

    Note:
    - sim = (a . b) / |a||b| where `a` and `b` are embedding vectors.
    """

    # TODO
    embedding_vectors: torch.Tensor = embedding.weight
    valid_indices: torch.Tensor = torch.randperm(valid_window, device=device)[:valid_size]

    valid_examples: torch.Tensor = embedding_vectors[valid_indices]

    dot_product: torch.Tensor = torch.mm(valid_examples, embedding_vectors.t())
    norms_valid: torch.Tensor = torch.norm(valid_examples, dim=1).unsqueeze(1)
    norms_embed: torch.Tensor = torch.norm(embedding_vectors, dim=1).unsqueeze(0)
    similarities: torch.Tensor = dot_product / (norms_valid * norms_embed)

    return valid_indices, similarities
