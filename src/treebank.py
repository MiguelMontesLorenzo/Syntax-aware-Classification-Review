from typing import List, Optional, Dict


class Node:
    def __init__(self, label: int, word: Optional[str] = None, parent=None) -> None:
        """
        Initialize Node class.

        Args:
        - label (int): sentiment label
        - word (str): word in the sentence
        - parent (Optional[Node]): node parent

        Returns:
        - None
        """
        self.label: int = label
        self.word: Optional[str] = word
        self.parent: Optional[Node] = parent
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.isLeaf: bool = False

    def __str__(self) -> str:
        """
        Print the string ofof the node and it's children.

        Args:
        - None

        Returns:
        - None
        """
        if self.isLeaf:
            return "[{0}:{1}]".format(self.word, self.label)
        return "({0} <- [{1}:{2}] -> {3})".format(
            self.left, self.word, self.label, self.right
        )


class Tree:
    def __init__(
        self, treeString: str, openChar: str = "(", closeChar: str = ")"
    ) -> None:
        """
        Initialize Tree class.

        Args:
        - treeString (str): string extracted from the dataset, is equivalent
            to a tree.
        - openChar (str): string that delimits the start of a constituyent.
        - closeChar (str): string that delimits the start of a constituyent.

        Returns:
        - None
        """
        tokens: List = []
        self.open: str = openChar
        self.close: str = closeChar

        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root: Node = self.parse(tokens)

        self.labels: List[Optional(int)] = self.get_labels(self.root)
        self.num_words: int = len(self.labels)

    def parse(self, tokens: List[str], parent: Optional[Node] = None) -> Node:
        """
        Parse a list of tokens (equivalent) to a sentence (or a subsentence)
        to convert the complete sentence to a tree structure.

        Args:
        - tokens (List[str]): list of tokens to parse.
        - parent (Optional[Node]):
            Node to start parsing,  if None, is the complete sentence.

        Returns:
        - node (Node): Node corresponding to the parent of the tree.
        """
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split: int = 2
        countOpen: int = 0
        countClose: int = 0

        if tokens[split] == self.open:
            countOpen += 1
            split += 1

        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node: Node = Node(int(tokens[1]), parent=parent)

        # Leaf Node
        if countOpen == 0:
            node.word = clean_sentence("".join(tokens[2:-1]).lower())
            node.isLeaf = True
            return node

        # Continue parsing the children
        node.left = self.parse(tokens[2:split], parent=node)
        node.right = self.parse(tokens[split:-1], parent=node)

        return node

    def get_words(self) -> List[Optional[str]]:
        """
        Get the words of the sentence to parse

        Args:
        - None

        Returns:
        - words (List[str]): list of words corresponding to the sentence
        """
        leaves: List[Optional[Node]] = self.get_leaves(self.root)
        words: List[Optional[str]] = [node.word for node in leaves if node]
        return words

    def get_labels(self, node: Optional[Node]) -> List[Optional[int]]:
        """
        Gets the terminal labels of the sentence starting with the node.

        Args:
        - node (Node): starting node

        Returns:
        - list of labels (List[int])
        """
        if node is None:
            return []
        return self.get_labels(node.left) + self.get_labels(node.right) + [node.label]

    def get_leaves(self, node: Node) -> List[Optional[Node]]:
        """
        Gets the terminal nodes of the sentence starting with the node.

        Args:
        - node (Node): starting node

        Returns:
        - list of nodes (List[Node])
        """
        if node is None:
            return []
        if node.isLeaf:
            return [node]
        else:
            return self.get_leaves(node.left) + self.get_leaves(node.right)


def clean_sentence(sentence: str) -> str:
    """
    Replaces incorrectly formatted characters.

    Args:
    - sentence (str): sentence to be cleaned.

    Returns:
    - sentence (str): cleaned sentence.
    """

    substitutions: Dict[str, str] = {"``": "", "''": ""}

    for key, value in substitutions.items():
        sentence = sentence.replace(key, value)
    return sentence.lower()
