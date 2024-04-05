from typing import List, Tuple, Dict, Set
import re


class Node:
    """
    Represents a node in the PCFG.
    """

    def __init__(self, father: str, children: List[str]) -> None:
        """
        Initialize Node with its father and children.

        Args:
        - father (str): The non-terminal symbol.
        - children (List[str]): The list of children nodes.

        Returns:
        - None
        """
        self.father: str = father
        self.children: List[str] = children
        self.count: int = 1
        self.probability: float = 0.0

    def __str__(self) -> str:
        return f"{self.father}: {self.children}"

    def compute_probability(self, fathers: Dict[str, int]) -> None:
        """
        Compute the probability of the node.

        Args:
        - Father (Dict[str, int]):

        Returns:
        - None
        """

        self.probability = self.count / fathers[self.father]

    def get_probability(self) -> float:
        """
        Get the probability of the node.

        Args:
        - None

        Returns:
        - float: The probability of the node.
        """
        return self.probability


class PCFG:
    """
    Represents a Probabilistic Context-Free Grammar.
    """

    def __init__(self) -> None:
        """
        Initialize the PCFG class.

        Args:
        - None

        Returns:
        - None
        """
        # Dictionary with the probability for each rule
        self.probability_rules: Dict[Tuple[str, Tuple[str, ...]], float] = {}

        # Dictionary to convert from tuple to Node
        self.nodes: Dict[Tuple[str, Tuple[str, ...]], Node] = {}

        # Dictionary to keep the count of each father
        self.fathers: Dict[str, int] = {}

        # List of non terminal nodes
        self.non_terminal: Set = set()

        # List of terminal nodes
        self.terminal: Set = set()

        # word2tag dictionary
        self.word2tag: Set = dict()

    def read_rules(self, path: str) -> None:
        """
        Read grammar rules from a file and convert them to CNF.

        Args:
        - path (str): The path to the file containing grammar rules.

        Returns:
        - None
        """
        with open(path, "r") as file:
            rules: List[str] = file.readlines()

        for rule in rules:
            rule = re.sub(r"[,.\n]", "", rule)
            rule_clean: str = re.sub(r"-\d+", "", rule)
            rule_clean = re.sub(r"=\d+", "", rule_clean)
            rule_clean_spaces: str = rule.replace(" ", "")
            derivation: List[str] = rule.split("->")

            if (
                rule.strip() != "->"
                and rule.strip() != ""
                and rule_clean_spaces != ""
                and derivation[0].strip() != ""
            ):
                self.convert_to_CNF(rule_clean)

    def convert_to_CNF(self, rule) -> None:
        """
        Convert a grammar rule to Chomsky Normal Form (CNF).

        Args:
        - rule (str): The grammar rule to convert.

        Returns:
        - None
        """

        derivation: Tuple[str] = rule.split("->")

        # Update or initialize the count for the father
        if derivation[0].strip() in self.fathers.keys():
            self.fathers[derivation[0].strip()] += 1
        else:
            self.fathers[derivation[0].strip()] = 1

        father = derivation[0].strip()

        # Obtain the children from the rule
        children = derivation[1].strip().split()

        if len(children) > 0 and len(children) <= 2:

            # Update the node count
            if (father, tuple(children)) not in self.nodes.keys():
                self.nodes[(father, tuple(children))] = Node(father, children)
            else:
                node = self.nodes[(father, tuple(children))]
                node.count += 1

            # Add to the terminal and non terminal nodes list
            if len(children) == 1:
                self.terminal.add(children[0].strip())
                if children[0].strip() not in self.word2tag.keys():
                    self.word2tag[children[0].strip()] = father
            else:
                for child in children:
                    self.non_terminal.add(child.strip())
            self.non_terminal.add(father)

        else:
            # Convert to CNF
            ...

    def compute_probabilities(self) -> None:
        """
        Compute probabilities for each node in the PCFG.

        Args:
        - None

        Returns:
        - None
        """
        for key, node in self.nodes.items():
            # Update probability
            node.compute_probability(self.fathers)

            # Storage the probability in the dictionary
            prob: float = node.get_probability()
            self.probability_rules[key] = prob
            # print(f"{str(node)} :: {str(prob)}")
