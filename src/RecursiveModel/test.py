from typing import List

from src.data import load_trees
from src.treebank import Tree

if __name__ == "__main__":
    train_data: List[Tree] = load_trees("train")
    labels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for tree in train_data:
        labels[tree.labels[-1]] += 1
    print(labels)
    # print(train_data[39].get_words())
    # print(train_data[39].labels)
