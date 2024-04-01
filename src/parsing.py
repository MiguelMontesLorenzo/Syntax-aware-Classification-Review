from src.grammar_conversion import PCFG, Father, Node
import torch


class CKY:
    def __init__(self, pcfg, input, device="cpu"):
        self.grammar: PCFG = pcfg

        self.input = input.lower().split()
        self.n = len(self.input)
        self.device = device

        self.tag2index = self.create_tag2index()

    def create_tag2index(self):

        return {tag: i for i, tag in enumerate(self.grammar.non_terminal)}

    def compute(self):

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
                j = i + l
                for X in self.grammar.non_terminal:
                    max_prob = 0
                    argmax_split = 0

                    for s in range(i, j):
                        for rule in self.grammar.nodes:

                            father, children = rule

                            if str(father) == X and len(children) > 1:
                                (Y, Z) = children
                                prob = (
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

    def build_tree(self, i, j, symbol, s_table, pi_table):

        if i == j:
            return f"Symbol: {symbol} Kid: {self.input[i]}"

        split = int(s_table[i, j, self.tag2index[symbol]].item())
        # Get the left and right symbols
        left_symbol_index = torch.argmax(pi_table[i, split, :]).item()
        right_symbol_index = torch.argmax(pi_table[split + 1, j, :]).item()
        left_symbol = list(self.tag2index.keys())[left_symbol_index]
        right_symbol = list(self.tag2index.keys())[right_symbol_index]

        left_child = self.build_tree(i, split, left_symbol, s_table, pi_table)
        right_child = self.build_tree(split + 1, j, right_symbol, s_table, pi_table)

        tree_str = f"Father: {symbol}\n"

        tree_str += "|- Left: " + left_child.replace("\n", "\n|  ") + "\n"
        tree_str += "|- Right: " + right_child.replace("\n", "\n|  ")

        return tree_str

    def parse(self):
        pi_table, s_table = self.compute()
        # Define the root node
        root_symbol = "S"

        # Build parse tree recursively
        print("Building tree...")
        parse_tree = self.build_tree(0, self.n - 1, root_symbol, s_table, pi_table)
        return parse_tree
