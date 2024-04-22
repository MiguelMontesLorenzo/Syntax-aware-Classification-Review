from typing import Dict, Optional
import torch
from torch import nn

from src.pretrained_embeddings import SkipGramNeg


class RNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) model implemented using PyTorch for sentiment
    classification.

    This model utilizes an embedding layer with pre-trained weights, followed
    by an LSTM layer
    for processing sequential data, and a linear layer for classification.
    The LSTM can be bidirectional or unidirectional.

    Attributes:
    - embedding (nn.Embedding): Embedding layer initialized with pre-trained weights.
    - embedding_dim (int): dimensions of embeddings.
    - bidirectional (bool): indicates whether it is bidirectional or not.
    - hidden_dim (int): size of the hidden state of the LSTM
    - rnn (nn.LSTM): LSTM (Long Short Term Memory) layer for
    processing sequential data.
    - fc (nn.Linear): Linear layer for classification.

    Args:
    - pretrained_model (SkipGramNeg): Pre-trained embedding model.
    - hidden_dim (int): the number of features in the hidden state of
    the LSTM.
    - num_classes (int): number of classes to classify.
    - num_layers (int): the number of layers in the LSTM.
    - bidirectional (bool): bidirectional or not.
    """

    def __init__(
        self,
        word2index: Dict[str, int],
        hidden_dim: int,
        embedding_dim: int,
        num_classes: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        device: str = "cpu",
        pretrained_model: Optional[SkipGramNeg] = None,
    ) -> None:
        """
        Initializes the RNN model with given embedding weights,
        hidden dimension, and number of layers.

         Args:
        - pretrained_model (SkipGramNeg): Pre-trained embedding
        model.
        - hidden_dim (int): the number of features in the hidden state
        of the LSTM.
        - num_classes (int): number of classes to classify.
        - num_layers (int): the number of layers in the LSTM.
        - bidirectional (bool): bidirectional or not.
        """

        super().__init__()
        if pretrained_model:
            self.embedding = pretrained_model.in_embed
            self.our_embeddings: bool = True

        else:
            self.embedding = nn.Embedding(len(word2index), embedding_dim).to(device)
            self.our_embeddings = False

        self.embedding_dim: int = embedding_dim
        self.bidirectional: bool = bidirectional
        self.hidden_dim: int = hidden_dim

        self.device: str = device

        self.rnn: nn.LSTM = nn.LSTM(
            self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.fc: nn.Linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the RNN model.

        Args:
        - x (torch.Tensor): The input tensor containing word indices.
        - text_lengths (torch.Tensor): Tensor containing the lengths of
        texts in the batch.

        Returns:
        - outputs (torch.Tensor): The output tensor after passing through
        the model.
        """
        embedded: torch.Tensor = self.embedding(x)

        packed_embedded: torch.Tensor = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu().numpy(), batch_first=True
        )

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        hidden = hidden[-1]

        outputs: torch.Tensor = self.fc(hidden).squeeze()
        return outputs
