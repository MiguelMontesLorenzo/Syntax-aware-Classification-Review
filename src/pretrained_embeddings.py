import torch
from torch import nn


class SkipGramNeg___(nn.Module):
    """
    A SkipGram model with Negative Sampling.

    This module implements a SkipGram model using negative sampling. It includes
    embedding layers for input and output words and initializes these embeddings
    with a uniform distribution to aid in convergence.

    Attributes:
    - vocab_size (int): integer count of the vocabulary size.
    - embed_dim (int): integer specifying the dimensionality of the embeddings.
    - noise_dist (torch.Tensor): tensor representing the distribution of noise words.
    - in_embed (nn.Embedding): embedding layer for input words.
    - out_embed (nn.Embedding): embedding layer for output words.
    """

    def __init__(self, vocab_size: int, embed_dim: int, noise_dist: torch.Tensor = None):
        """
        Initializes the SkipGramNeg model with given vocabulary size, embedding size, and noise distribution.

        Args:
        - vocab_size (int): size of the vocabulary.
        - embed_dim (int): size of each embedding vector.
        - noise_dist (torch.Tensor): distribution of noise words for negative sampling.
        """
        super().__init__()
        self.vocab_size: int = vocab_size
        self.embed_dim: int = embed_dim
        self.noise_dist: torch.Tensor = noise_dist

        # Define embedding layers for input and output words
        self.in_embed: nn.Embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.out_embed: nn.Embedding = nn.Embedding(self.vocab_size, self.embed_dim)
    
    def forward(self, inputs):
        return self.in_embed(inputs)


class SkipGramNeg(nn.Module):
    """
    A SkipGram model with Negative Sampling.
    """

    def __init__(self, vocab_size: int, embed_dim: int, noise_dist: torch.Tensor = None):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.embed_dim: int = embed_dim
        self.noise_dist: torch.Tensor = noise_dist

        # Define embedding layers for input and output words
        self.in_embed: nn.Embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.out_embed: nn.Embedding = nn.Embedding(self.vocab_size, self.embed_dim)

    def load_pretrained_embeddings(self, pretrained_weights):
        """
        Load pre-trained embeddings into the in_embed layer.

        Args:
        - pretrained_weights (torch.Tensor): Pre-trained embedding weights to be loaded.
        """
        # Truncate the weights to match the size of the current vocabulary
        truncated_weights = pretrained_weights[:self.vocab_size, :]

        # Assign the truncated weights to the in_embed layer
        self.in_embed.weight.data.copy_(truncated_weights)

    def forward(self, inputs):
        return self.in_embed(inputs)