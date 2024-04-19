from typing import List, Dict, Tuple

from src.treebank import Tree
from src.RecursiveModel.utils import flatten


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
