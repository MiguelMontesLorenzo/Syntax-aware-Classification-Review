from typing import List, Tuple, Dict, Set
from collections import Counter
import torch
import random
import spacy
import re
import csv
import subprocess

from math import sqrt
from torch.utils.data import Dataset, DataLoader

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(txt_infile: str, csv_infile: str) -> Tuple[List[str], List[str], Dict[int, str]]:
    """
    Load text and csv data and preprocess it using a tokenize function.

    Args:
    - txt_infile (str): Path to the input txt file containing text data form Penn Treebank movie reviews.
    - csv_infile (str): Path to the input csv file containing data form IMBD movie reviews.

    Returns:
    - sentences (List[str]): list of correctly processed sentences fomr both files.
    - tokens (List[str]): list of preprocessed and tokenized words from the input data.
    - correspondences (Dict[int, Tuple(int)]): association of word index in tokens and the index of the sentence that
    word belongs to.
    """
    sentences, text = preprocess_text(txt_infile, csv_infile)
    tokens, correspondences = tokenize(sentences, text)

    return sentences, tokens, correspondences


def preprocess_text(txt_infile: str, csv_infile: str) -> Tuple[List[str], str]:
    """
    Reads the csv and txt files with movie reviews and process them to extract the individual sentences
    (where dependency parsing can be donde correctly).

    Args:
    - txt_infile (str): Path to the input txt file containing text data form Penn Treebank movie reviews.
    - csv_infile (str): Path to the input csv file containing data form IMBD movie reviews.

    Returns:
    - sentences (List[str]): list of correctly processed sentences fomr both files.
    """
    encoding: str = "utf-8"
    sentences: list = []
    whole_text: str = ""
    padding: str = " <padding> "

    with open(txt_infile, "r", encoding=encoding) as file:
        text: List[str] = file.readlines()
        new_text: List[str] = remove_index_elements(text)

        for elem in new_text:
            cleaned_sentence = clean_sentence(elem)
            splitted_sentences: List[str] = split_sentence(cleaned_sentence)
            filtered_sentences: List[str] = filter_sentence(splitted_sentences)

            if len(filtered_sentences) > 0:
                sentences.extend(filtered_sentences)
            
            whole_text += " ".join(filtered_sentences)
            whole_text += padding
    
    with open(csv_infile, "r", newline="", encoding=encoding) as csv_file:
        csv_reader = csv.reader(csv_file)

        # The second element is the score, irrelevant for the task
        first_sentences: List[str] = [review[0] for review in csv_reader]

        for elem in first_sentences:
            cleaned_sentence = clean_sentence(elem)
            splitted_sentences: List[str] = split_sentence(cleaned_sentence)
            filtered_sentences: List[str] = filter_sentence(splitted_sentences)

            if len(filtered_sentences) > 0:
                sentences.extend(filtered_sentences)
            
            whole_text += " ".join(filtered_sentences)
            whole_text += padding
        return sentences, whole_text


def remove_index_elements(sentences: List[str]) -> List[str]:
    """
    Removes non-necessary elements from txt sentences.
    
    Args:
    - sentences (List[str]): sentences to be cleaned.

    Returns:
    - processed_sentences (List[str]): cleaned sentences.
    """
    processed_sentences: List[str] = []

    for line in sentences:
        parts: List[str] = line.split('\t')

        if len(parts) > 1:
            sentence = parts[1]
            cleaned_sentence: str = sentence.rstrip('\n')
            processed_sentences.append(cleaned_sentence)
    
    return processed_sentences


def clean_sentence(sentence: str) -> str:
    """
    Replaces incorrectly formatted characters.
    
    Args:
    - sentence (str): sentence to be cleaned.

    Returns:
    - sentence (str): cleaned sentence.
    """
    substitutions: Dict[str, str] = {
        " '": "'",
        "`` ": "",
        "''": "",
        "` ": "",
        "' ": " ",
        "<br /><br />": " ",
        "\'s": "'s"
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
    - new_sentence_list (List[str]): sentences splitted from the original one.
    """
    splitters: List[str] = [" and ", ".", ",", ";", "--", ":", "!", "?"]
    previous_sentence_list: List[str] = [sentence]

    for splitter in splitters:
        new_sentence_list: List[str] = []
        for sentence in previous_sentence_list:
            new_sentence_list.extend([splitted.strip() for splitted in sentence.split(splitter)])
        previous_sentence_list = new_sentence_list

    return new_sentence_list


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


def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates lookup tables for vocabulary.

    Args:
    - words (List[str]): A list of words from which to create vocabulary.

    Returns:
    - vocab_to_int (Dict[str, int]): dictionary that maps words to integers.
    - vocab_to_int (Dict[int, str]): dictionary that maps integers to words.
    """

    word_counts: Counter = Counter(words)

    sorted_vocab: List[str] = sorted(word_counts, key=word_counts.get, reverse=True)
    
    vocab_to_int: Dict[str, int] = {word: i for i, word in enumerate(sorted_vocab)}
    int_to_vocab: Dict[str, int] = {i: word for i, word in enumerate(sorted_vocab)}

    return vocab_to_int, int_to_vocab


def subsample_words(words: List[str], vocab_to_int: Dict[str, int], correspondences: Dict[int, str], threshold: float = 1.0) -> Tuple[List[int], Dict[str, float], Dict[int, str]]:
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
    int_words: List[int] = [vocab_to_int[word] for word in words]

    n_words: int = len(words)
    freqs: Dict[str, float] = {word: freq/n_words for (word, freq) in Counter(words).items()}
    train_words: List[int] = []

    index: int = 0
    for i, word in enumerate(words):
        if random.random() < sqrt(threshold/freqs[word]):
            train_words.append(int_words[i])
            sampled_correspondences[index] = correspondences[i]
            index += 1

    return train_words, freqs, sampled_correspondences


def get_neighbours(tree: spacy.tokens.doc.Doc, idx: int) -> List[str]:
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
    # print("Target", target)
    neighbours: Set[str] = set()

    for token in tree:
        if token != target and (token.head == target or target.head == token) and token.dep_ not in ["det", "prep"]:
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
    # print("Neighbour", neighbours)
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
    
    target_words: Set[str] = set()

    neighbours: List[str] = get_neighbours(dependency_tree, word_idx)
    for neighbour in neighbours:
        neighbour: str = re.sub(r"[^a-zA-Z]", "", neighbour)
        if neighbour in vocab_to_int.keys():
            target_words.add(vocab_to_int[neighbour])

    n_words: int = len(words)
    window_size: int = random.randint(1, window_size)
    target_words_print = []

    for i in range(1, window_size + 1):
        if idx - i >= 0 and words[idx - i] != "padding":
            target_words.add(words[idx - i])
            target_words_print.append(int_to_vocab[words[idx - i]])
        if idx + i < n_words and words[idx + i] != "padding":
            target_words.add(words[idx + i])
            target_words_print.append(int_to_vocab[words[idx + i]])
    
    target_words: List[str] = list(target_words)
    # print("Nearby words", target_words_print)
    
    while len(target_words) < context_length:
        target_words.append(vocab_to_int["padding"])
    
    target_words = target_words[:context_length]

    return target_words


class EmbeddingsDataset(Dataset):
    def __init__(self, tokens: List[str], sentences: List[str], correspondences: Dict[int, str], nlp: spacy.language.Language, vocab_to_int: Dict[str, int], int_to_vocab: Dict[int, str], window_size: int = 3) -> None:
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
        # print(idx, targets)
        # print()

        targets_tensor: torch.Tensor = torch.tensor(new_targets)
        
        return input_tensor, targets_tensor


def generate_data_loader(tokens: list[int], sentences: List[str], correspondences: (Dict[int, str]), batch_size: int, vocab_to_int: Dict[str, int], int_to_vocab: Dict[int, str]):
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
