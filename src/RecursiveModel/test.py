from src.RecursiveModel.data import download_data, load_trees, load_vocab
from src.RecursiveModel.recursive import RNTN
from src.RecursiveModel.train_functions import train, val, test
from src.RecursiveModel.treebank import Tree
from src.RecursiveModel.utils import get_batch, flatten, save_model

from typing import List

if __name__ == "__main__":
    train_data: List[Tree] = load_trees("train")
    labels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for tree in train_data:
        labels[tree.labels[-1]] += 1
    print(labels)
