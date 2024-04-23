# deep learning libraries
import torch
from src.pretrained_embeddings import SkipGramNeg
from abc import abstractmethod


class Weighter:
    def __init__(self, vocab_size=2, output_size=2) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.output_size: int = output_size

        return None

    @abstractmethod
    def fit(self, sentences: list[list[int]], labels: list[int]) -> None:
        pass

    @abstractmethod
    def weight_words(self, sentence: torch.Tensor) -> torch.Tensor:
        pass


class UniformWeighter(Weighter):
    def __init__(self, vocab_size=2, output_size=2) -> None:
        super().__init__(vocab_size, output_size)
        return None

    def fit(self, sentences: list[list[int]], labels: list[int]) -> None:
        pass

    def weight_words(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Method to weght words uniformly based on the input sentence.

        Args:
            senetence (torch.Tensor): The input sentence
        Returns:
            torch.Tensor: The weights for each word in the sentence
        """

        # compute weights
        unnormalized_weights: torch.Tensor = torch.ones_like(sentence)
        normalized_weights: torch.Tensor = unnormalized_weights / torch.sum(
            unnormalized_weights, dim=1
        )

        return normalized_weights


class NaiveBayesWeighter(Weighter):
    def __init__(self, vocab_size=2, output_size=2, alpha=1) -> None:
        super().__init__(vocab_size, output_size)

        """
        VecAvg Weighter that uses Naive Bayes to compute weights.
        Implementation based on following paper: 
            A Text Classifier Using Weighted Average Word Embedding
        """

        self._alpha: float = alpha

        # Initialize the counters for each class
        self.class_word_counts: torch.Tensor = torch.zeros(
            (output_size, self.vocab_size)
        )
        self.class_counts: torch.Tensor = torch.zeros(output_size)

        return

    def fit(self, sentences: list[list[int]], labels: list[int]) -> None:
        """
        Precomputes the class log ratios for each word

        Args:
            sentences (list[torch.Tensor]): list of input sentences
        Returns:
            torch.Tensor = torch.zeros((len(sentences), self.
        """

        for sentence, label in zip(sentences, labels):
            for word in sentence:
                self.class_word_counts[label, word] += 1
            self.class_counts[label] += 1

        self.r: torch.Tensor = torch.zeros((self.output_size, self.vocab_size))
        for i in range(self.output_size):
            p: torch.Tensor = self._alpha + self.class_word_counts[i]
            q: torch.Tensor = self._alpha + (
                self.class_word_counts.sum(0) - self.class_word_counts[i]
            )
            self.r[i] = torch.log(
                (p / torch.norm(p, 1)) / (q / torch.norm(q, 1))
            ).float()

    def weight_words(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Method to weght words using the precomputed log ratios

        Args:
            senetence (torch.Tensor): The input sentence
        Returns:
            torch.Tensor: The weights for each word in the sentence
        """
        weights: torch.Tensor = torch.zeros_like(sentence)
        weights, _ = torch.max(self.r[:, sentence], dim=0)
        weights = torch.sigmoid(weights)
        weights = weights - torch.min(weights, dim=0)[0]
        weights = weights / torch.sum(weights, dim=1)[:, None]

        return weights


class VecAvg(torch.nn.Module):
    """
    This is the class to construct the model. Only layers defined in
    this script can be used.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: tuple[int, ...],
        pretrained_model: SkipGramNeg,
    ) -> None:
        """
        This method is the constructor of the model.

        Args:
            input_size: size of the input
            output_size: size of the output
            hidden_sizes: three hidden sizes of the model
        """
        super().__init__()

        # define embeddings
        self.embeddings: torch.nn.Module = pretrained_model.in_embed

        layers: list = []
        layer_sizes: list = [input_size, *hidden_sizes, output_size]

        # Añadir capas
        for i, _ in enumerate(layer_sizes[:-1]):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.classifier = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, weighter: Weighter) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: input tensor, Dimensions: [batch, channels, height,
                width].
            weighter: Weighter object to compute word weights.

        Returns:
            outputs of the model. Dimensions: [batch, 1].
        """

        embedded_inputs: torch.Tensor = self.embeddings(inputs)
        weights: torch.Tensor = weighter.weight_words(inputs)
        weighted_embedded_inputs: torch.Tensor = torch.einsum(
            "ij,ijk->ik", weights, embedded_inputs
        )
        outputs: torch.Tensor = self.classifier(weighted_embedded_inputs)
        outputs = torch.softmax(outputs, dim=1)

        return outputs
