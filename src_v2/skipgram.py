import torch
from torch import nn

class SkipGramNeg(nn.Module):
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

        # Initialize embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)


    def forward_input(self, input_words: torch.Tensor) -> torch.Tensor:
        """
        Fetches input vectors for a batch of input words.

        Args:
        - input_words (torch.Tensor): tensor of integers representing input words.

        Returns:
        - input_vectors (torch.Tensor): tensor containing the input vectors for the given words.
        """
        return self.in_embed(input_words)

    def forward_output(self, output_words: torch.Tensor) -> torch.Tensor:
        """
        Fetches output vectors for a batch of output words.

        Args:
        - output_words (torch.Tensor): tensor of integers representing output words.

        Returns:
        - output_vectors (torch.Tensor): tensor containing the output vectors for the given words.
        """
        return self.out_embed(output_words)

    def forward_noise(self, batch_size: int, n_samples: int) -> torch.Tensor:
        """
        Generates noise vectors for negative sampling.

        Args:
        - batch_size (int): number of words in each batch.
        - n_samples (int): number of negative samples to generate per word.

        Returns:
        - noise_vectors (torch.Tensor): tensor of noise vectors with shape (batch_size, n_samples, n_embed).
        """
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist: torch.Tensor = torch.ones(self.vocab_size)
        else:
            noise_dist: torch.Tensor = self.noise_dist

        # Sample words from our noise distribution
        noise_words: torch.Tensor = noise_dist.multinomial(num_samples=batch_size * n_samples, replacement=True)

        device: str = "cuda" if self.out_embed.weight.is_cuda else "cpu"
        noise_words: torch.Tensor = noise_words.to(device)

        # Reshape output vectors to size (batch_size, n_samples, n_embed)
        noise_vectors: torch.Tensor = self.out_embed(noise_words)

        noise_vectors: torch.Tensor = noise_vectors.reshape(batch_size, n_samples, self.embed_dim)

        return noise_vectors

    
class NegativeSamplingLoss(nn.Module):
    """
    Implements the Negative Sampling loss as a PyTorch module.

    This loss is used for training word embedding models like Word2Vec using
    negative sampling. It computes the loss as the sum of the log-sigmoid of
    the dot product of input and output vectors (for positive samples) and the
    log-sigmoid of the dot product of input vectors and noise vectors (for
    negative samples), across a batch.
    """

    def __init__(self):
        """Initializes the NegativeSamplingLoss module."""
        super().__init__()

    def forward(self, input_vectors: torch.Tensor, output_vectors: torch.Tensor,
                noise_vectors: torch.Tensor) -> torch.Tensor:
        """Computes the Negative Sampling loss.

        Args:
        - input_vectors (torch.Tensor): A tensor containing input word vectors, 
                            shape (batch_size, embed_size).
        - output_vectors (torch.Tensor): A tensor containing output word vectors (positive samples), 
                            shape (batch_size, embed_size).
        - noise_vectors (torch.Tensor): A tensor containing vectors for negative samples, 
                            shape (batch_size, n_samples, embed_size).

        Returns:
        - total (torch.Tensor): tensor containing the average loss for the batch.
        """
        input_vectors: torch.Tensor = input_vectors.reshape(input_vectors.size(0), input_vectors.size(1), 1)
        output_vectors: torch.Tensor = output_vectors.reshape(input_vectors.size(0), 1, input_vectors.size(1))
        
        # Compute log-sigmoid loss for correct classifications
        out_loss: torch.Tensor= nn.functional.logsigmoid(torch.bmm(output_vectors, input_vectors))

        # Compute log-sigmoid loss for incorrect classifications
        noise_loss: torch.Tensor = nn.functional.logsigmoid(torch.bmm(-noise_vectors, input_vectors))

        # Return the negative sum of the correct and noisy log-sigmoid losses, averaged over the batch
        total: torch.Tensor = (-out_loss - noise_loss.squeeze().sum()).mean()
        return total