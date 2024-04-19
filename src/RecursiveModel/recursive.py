import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Dict, List

from src.treebank import Node, Tree
from src.pretrained_embeddings import SkipGramNeg


class RNTN(nn.Module):
    def __init__(
        self,
        word2index: Dict[str, int],
        hidden_size: int,
        output_size: int,
        simple_RNN: bool = False,
        device: str = "cpu",
        pretrained_model: SkipGramNeg =None,
    ) -> None:
        """
        Construct the recursive NN.

        Args:
        - word2index (Dict[str, int]): Dictionary that maps each vocab word to an index.
        - hidden_size (int)
        - output_size (int)
        - simple_RNN (bool): RNTN or simple RNN.
        - device (str)

        Returns:
        - None
        """
        super().__init__()
        self.word2index: Dict[str, int] = word2index

        if pretrained_model:
            self.embed: torch.Tensor = pretrained_model.in_embed
            self.initialize_embeddings: bool = False

        else:
            self.embed: nn.Embedding = nn.Embedding(len(word2index), hidden_size).to(
                device
            )
            self.initialize_embeddings: bool = True

        self.V: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size * 2, hidden_size * 2))
                for _ in range(hidden_size)
            ]
        )
        self.W: nn.Parameter = nn.Parameter(torch.randn(hidden_size * 2, hidden_size))
        self.b: nn.Parameter = nn.Parameter(torch.randn(1, hidden_size))
        self.W_out: nn.Parameter = nn.Parameter(torch.randn(hidden_size, output_size))
        self.simple_RNN: bool = simple_RNN
        self.device: str = device
        self.output_size: int = output_size

    def init_weight(self) -> None:
        """
        Initialize all the parameters following Xavier's initialization.

        Args:
        - None

        Returns:
        - None
        """

        # Embeddings
        if self.initialize_embeddings:
            nn.init.xavier_uniform_(self.embed.state_dict()["weight"])

        # V
        if self.simple_RNN:
            self.V = torch.zeros_like(self.V, dtype=self.V.dtype, device=self.device)
        else:
            for param in self.V.parameters():
                nn.init.xavier_uniform_(param)

        # Weights and bias
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.W_out)
        nn.init.xavier_uniform_(self.b)

    def tree_propagation(self, node: Node) -> Dict[Node, torch.Tensor]:
        """
        Compute the propagation of a node following the equation:
        h = f(h + Wx + b)

        Args:
        - node (Node): node to propagate from

        Returns:
        - recursive_tensor (Dict[Node, int]): Dictionary with the node and it's vector representation
        """
        recursive_tensor: Dict[Node, torch.Tensor] = OrderedDict()

        # Get the vector representation of the current node
        if node.isLeaf:
            if node.word in self.word2index.keys():
                word_vector: Variable = Variable(
                    torch.tensor(
                        [self.word2index[node.word]],
                        dtype=torch.long,
                        device=self.device,
                    )
                )
            else:
                word_vector: Variable = Variable(
                    torch.tensor(
                        [self.word2index["<UNK>"]], dtype=torch.long, device=self.device
                    )
                )

            current_vector = self.embed(word_vector)

        else:
            # Update the chidren
            recursive_tensor.update(self.tree_propagation(node.left))
            recursive_tensor.update(self.tree_propagation(node.right))

            children_stack: torch.Tensor = torch.cat(
                [recursive_tensor[node.left], recursive_tensor[node.right]],
                1,
            )

            Wx: torch.Tensor = torch.matmul(children_stack, self.W)

            if self.simple_RNN:
                current_vector: torch.Tensor = F.tanh(Wx + self.b)

            else:
                # h: List = []
                # for v in self.V:
                #     h.append(
                #         torch.matmul(
                #             torch.matmul(children_stack, v),
                #             children_stack.transpose(0, 1),
                #         )
                #     )

                h: List = [
                    torch.matmul(
                        torch.matmul(children_stack, v), children_stack.transpose(0, 1)
                    )
                    for v in self.V
                ]
                h_tensor: torch.Tensor = torch.cat(h, 1)

                current_vector: torch.Tensor = F.tanh(h_tensor + Wx + self.b)

        recursive_tensor[node] = current_vector

        return recursive_tensor

    def forward(self, trees: List[Tree], root_only: bool = False) -> torch.Tensor:
        """
        Forward propagation of the net.

        Args:
        - trees (List[Tree]):

        Returns:
        - root_only (bool):
        """
        propagated: List = []

        for tree in trees:
            recursive_tensor: Dict[Node, torch.Tensor] = self.tree_propagation(
                tree.root
            )
            if root_only:
                recursive_tensor_element: torch.Tensor = recursive_tensor[tree.root]
                propagated.append(recursive_tensor_element)
            else:
                recursive_tensor_list: List[torch.Tensor] = [
                    tensor for _, tensor in recursive_tensor.items()
                ]
                propagated.extend(recursive_tensor_list)

        propagated: torch.Tensor = torch.cat(propagated)

        output: torch.Tensor = F.log_softmax(
            torch.matmul(propagated, self.W_out), dim=1
        )

        return output
