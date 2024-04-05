from typing import List, Tuple, Dict, Set
from src.grammar_conversion import PCFG, Node
import torch


class CKY:
    def __init__(self, pcfg: PCFG, input: str, device: str = "cpu") -> None:
        """
        Initialize the CKY class with the context free grammar and inputs.

        Args:
        - pcfg (PCFG): The probabilistic context free grammar calculated.
        - input (str): The sentence to be parsed.
        - device (str): The device for the tensors.

        Returns:
        - None
        """
        self.grammar: PCFG = pcfg

        self.input: List[str] = input.lower().split()
        self.n: int = len(self.input)
        self.device: str = device

        self.tag2index: Dict[str, int] = self.create_tag2index()

    def create_tag2index(self) -> Dict[str, int]:
        """
        Create a dictionary to associate each tag to a number.

        Args:
        - None

        Returns:
        - dictionary (Dict[str, int]): Association tag with number.
        """
        return {tag: i for i, tag in enumerate(self.grammar.non_terminal)}

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute complete CKY algorithm.

        Args:
        - None

        Returns:
        - pi_table (torch.Tensor): to save probabilities.
        - s_table (torch.Tensor): to save split points.
        """
        # Initialization
        pi_table: torch.Tensor = torch.zeros(
            (self.n, self.n, len(self.grammar.non_terminal)), device=self.device
        ).fill_(1e-50)

        s_table: torch.Tensor = torch.zeros(
            (self.n, self.n, len(self.grammar.non_terminal)), device=self.device
        )

        for i, word in enumerate(self.input):

            try:
                tag: str = self.grammar.word2tag[word]
                pi_table[i, i, self.tag2index[tag]] = self.grammar.probability_rules[
                    (tag, (word,))
                ]
            # Ignore the words that haven't appeared in the training
            except KeyError:
                pass

        # Iteration
        for l in range(1, self.n):
            for i in range(self.n - l):
                j: int = i + l
                for X in self.grammar.non_terminal:
                    max_prob: float = 0.0
                    argmax_split: int = 0

                    for s in range(i, j):
                        for rule in self.grammar.nodes:
                            father: str
                            children: str
                            father, children = rule

                            if father == X and len(children) > 1:
                                Y: str
                                Z: str
                                (Y, Z) = children
                                prob: float = (
                                    pi_table[i, s, self.tag2index[Y]].item()
                                    * pi_table[s + 1, j, self.tag2index[Z]].item()
                                    * self.grammar.probability_rules[rule]
                                )

                                if prob > max_prob:
                                    max_prob = prob
                                    argmax_split = s

                    pi_table[i, j, self.tag2index[X]] = max_prob
                    s_table[i, j, self.tag2index[X]] = argmax_split

        return pi_table, s_table

    def build_tree(
        self, i: int, j: int, symbol: str, s_table: torch.Tensor, pi_table: torch.Tensor
    ) -> str:
        """
        Recursive method to build the tree based on the pi and s tables.

        Args:
        - i (int): start point of the sentence.
        - j (int): end point of the sentence.
        - symbol (str): parent symbol.
        - s_table (torch.Tensor): to save split points.
        - pi_table (torch.Tensor): to save probabilities.

        Returns:
        - tree_str (str): parsed tree.
        """

        if i == j:
            return f"Symbol: {symbol.upper()} Kid: {self.input[i]}"

        split: int = int(s_table[i, j, self.tag2index[symbol]].item())
        # Get the left and right symbols
        left_symbol_index: int = torch.argmax(pi_table[i, split, :]).item()
        right_symbol_index: int = torch.argmax(pi_table[split + 1, j, :]).item()
        left_symbol: str = list(self.tag2index.keys())[left_symbol_index]
        right_symbol: str = list(self.tag2index.keys())[right_symbol_index]

        left_child: str = self.build_tree(i, split, left_symbol, s_table, pi_table)
        right_child: str = self.build_tree(
            split + 1, j, right_symbol, s_table, pi_table
        )

        tree_str: str = f"Father: {symbol.upper()}\n"

        tree_str += "|- Left: " + left_child.replace("\n", "\n|  ") + "\n"
        tree_str += "|- Right: " + right_child.replace("\n", "\n|  ")

        return tree_str

    def parse(self) -> str:
        """
        Execute CKY and correspoding tree.

        Args:
        - None

        Returns:
        - parse_tree (str): builded tree.
        """
        print("Constructing tables...")
        pi_table: torch.Tensor
        s_table: torch.Tensor
        pi_table, s_table = self.compute()
        # Define the root node
        root_symbol: str = "s"

        # Build parse tree recursively
        print("Building tree...")
        parse_tree: str = self.build_tree(0, self.n - 1, root_symbol, s_table, pi_table)
        return parse_tree
