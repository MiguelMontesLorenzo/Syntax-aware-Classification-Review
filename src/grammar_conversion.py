from typing import List, Tuple, Dict
import re


class Father:
    """
    Represents a non-terminal symbol in the grammar.
    """
    def __init__(self, identity: str) -> None:
        """
        Initialize Father with its identity.

        Args:
        - identity (str): The identity of the non-terminal symbol.
        """
        self.identity = identity
        self.count = 1

    def __str__(self) -> str:
        return self.identity


class Node:
    """
    Represents a node in the PCFG.
    """
    def __init__(self, father: Father, children: List[str]) -> None:
        """
        Initialize Node with its father and children.

        Args:
        - father (Father): The non-terminal symbol.
        - children (List[str]): The list of children nodes.
        """
        self.father = father
        self.children = children
        self.count = 1
        self.probability = 0

    def __str__(self) -> str:
        return f'{self.father}: {self.children}'

    def compute_probability(self) -> None:
        """
        Compute the probability of the node.
        """
        self.probability = self.count / self.father.count

    def get_probability(self) -> float:
        """
        Get the probability of the node.

        Returns:
        - float: The probability of the node.
        """
        return self.probability


class PCFG():
    """
    Represents a Probabilistic Context-Free Grammar.
    """
    def __init__(self) -> None:
        # Dictionary with the probability for each rule
        self.probability_rules: Dict[Tuple[Father, Tuple[str, ...]], float] = {}

        # Dictionary to convert from tuple to Node
        self.nodes: Dict[Tuple[Father, Tuple[str, ...]], Node] = {}

        # Dictionary to keep the count of each father
        self.fathers: Dict[str, int] = {}

        # List of non terminal nodes
        self.non_terminal = set()

        # List of terminal nodes
        self.terminal = set()

    def read_rules(self, path: str) -> None:
        """
        Read grammar rules from a file and convert them to CNF.

        Args:
        - path (str): The path to the file containing grammar rules.
        """
        with open(path, "r") as file:
            rules: List[str] = file.readlines()

        for rule in rules:
            rule: str = re.sub(r'[,.\n]', '', rule)
            if rule.strip() != "->" and rule.strip() != "":
                self.convert_to_CNF(rule)

    def convert_to_CNF(self, rule) -> None:
        """
        Convert a grammar rule to Chomsky Normal Form (CNF).

        Args:
        - rule (str): The grammar rule to convert.
        """

        derivation: Tuple[str] = rule.split("->")

        # Update or initialize the count for the father
        if derivation[0] in self.fathers.keys():
            self.fathers[derivation[0]] += 1
        else:
            self.fathers[derivation[0]] = 1

        father = Father(derivation[0])
        father.count = self.fathers[derivation[0]]

        # Obtain the children from the rule
        children = derivation[1].split()

        if len(children) <= 2:
            # Add to prob rules
            if (father, tuple(children)) not in self.nodes.keys():
                self.nodes[(father, tuple(children))] = Node(father, children)
            else:
                # Update the node count
                node = self.nodes[(father, tuple(children))]
                node.count += 1

            # Add to the terminal and non terminal nodes list
            if len(children) == 1:
                self.terminal.add(children[0])
            else:
                for child in children:
                    self.non_terminal.add(child)
            self.non_terminal.add(derivation[0])

        else:
            # Convert to CNF
            ...

    def compute_probabilities(self) -> None:
        """
        Compute probabilities for each node in the PCFG.
        """
        for key, node in self.nodes.items():
            # Update probability
            node.compute_probability()

            # Storage the probability in the dictionary
            prob: float = node.get_probability()
            self.probability_rules[key] = prob
            print(f"{str(node)} :: {str(prob)}")
