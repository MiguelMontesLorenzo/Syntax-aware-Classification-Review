from typing import List, Dict

from src.data import load_trees
from src.treebank import Tree

if __name__ == "__main__":
    train_data: List[Tree] = load_trees("test+")
    labels: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for tree in train_data:
        if tree.labels and tree.labels[-1] is not None:
            labels[tree.labels[-1]] += 1
    print(labels)
    # print(train_data[39].get_words())
    # print(train_data[39].labels)
