import nltk
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

import os
from typing import List, Tuple


def extract_rules(tree: nltk.Tree, rules: List) -> List[Tuple[str, str]]:
    """
    Recursively extract rules from an nltk.Tree object.

    Args:
    - tree: nltk.Tree
        The tree from which rules are to be extracted.
    - rules: list
        A list to store the extracted rules. Each rule is represented
        as a tuple (label, child_labels).

    Returns:
    - rules: list
        List of extracted rules.
    """

    if isinstance(tree, nltk.Tree):
        rules.append(
            (
                tree.label(),
                " ".join(
                    child.label() if isinstance(child, nltk.Tree) else child
                    for child in tree
                ),
            )
        )
        for child in tree:
            rules = extract_rules(child, rules)
    return rules


def load_data(path: str, filepath: str) -> None:
    """
    Load data from PRD files, extract grammar rules, and write them to a
    text file.

    This function downloads data from a specified path, reads PRD files,
    extracts grammar rules from parsed sentences, and writes the rules
    to a text file named 'grammar_rules.txt' in the same directory.

    Args:
    - None

    Returns:
    - None
    """
    # Download data
    download_data(path)

    # Initialize a BracketParseCorpusReader
    path_files = path + "/corpora/treebank/combined"

    prd_reader: BracketParseCorpusReader = BracketParseCorpusReader(
        path_files, r".*\.mrg"
    )

    # Get a list of file identifiers (fileids)
    fileids: List[str] = prd_reader.fileids()

    # Read the trees from the iles
    with open(filepath, "w") as file:
        for fileid in fileids:
            trees: List[nltk.Tree] = prd_reader.parsed_sents(fileid)
            for tree in trees:
                rules: List[Tuple[str, str]] = []
                rules = extract_rules(tree, rules)
                for rule in rules:
                    file.write(rule[0].lower() + " -> " + rule[1].lower() + "\n")
                file.write("\n")


def download_data(path: str) -> None:
    """
    Download the data from nltk and save it  in the given directory

    Args:
    - path: string with the path where the files will be downloaded

    Returns:
    - None
    """
    # Checking if provided directory exist and if not create it
    if not os.path.exists(path):
        os.makedirs(path)

        # Download the Penn Treebank corpus
        nltk.download("treebank", download_dir=path)
