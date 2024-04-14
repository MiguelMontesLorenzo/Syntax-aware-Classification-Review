import torch
from torch import nn
from src.pretrained_embeddings import SkipGramNeg


class RNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) model implemented using PyTorch for text classification.

    This model utilizes an embedding layer with pre-trained weights, followed by an LSTM layer
    for processing sequential data, and a linear layer for classification.

    Attributes:
        embedding (nn.Embedding): Embedding layer initialized with pre-trained weights.
        rnn (nn.LSTM): LSTM (Long Short Term Memory) layer for processing sequential data.
        fc (nn.Linear): Linear layer for classification.

    Args:
        embedding_weights (torch.Tensor): Pre-trained word embeddings.
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of layers in the LSTM.
    """

    def __init__(self, pretrained_model: SkipGramNeg, hidden_dim: int, num_layers: int, bidirectional: bool = True) -> None:
        """
        Initializes the RNN model with given embedding weights, hidden dimension, and number of layers.

        Args:
            embedding_weights (torch.Tensor): The pre-trained embedding weights to be used in the embedding layer.
            hidden_dim (int): The size of the hidden state in the LSTM layer.
            num_layers (int): The number of layers in the LSTM.
        """
        super().__init__()
        self.embedding = pretrained_model.in_embed
        self.embedding_dim = pretrained_model.embed_dim

        self.rnn: nn.LSTM = nn.LSTM(self.embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        self.fc: nn.Linear = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, x: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the RNN model.

        Args:
            x (torch.Tensor): The input tensor containing word indices.
            text_lengths (torch.Tensor): Tensor containing the lengths of texts in the batch.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        embedded: torch.Tensor = self.embedding(x)

        packed_embedded: torch.Tensor = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu().numpy(), batch_first=True)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        hidden: torch.Tensor = hidden[-1]

        outputs: torch.Tensor = self.fc(hidden).squeeze()
        return outputs