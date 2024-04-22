# deep learning libraries
import torch
from src.pretrained_embeddings import SkipGramNeg

# from exceptions import NotImplementedError


class Weighter:
    def __init__(self, vocabulary_size=2, output_size=2) -> None:
        super().__init__()
        self.vocab_size: int = vocabulary_size
        self.output_size: int = output_size

        return None


class UniformWeighter(Weighter):
    def __init__(self) -> None:
        super().__init__()
        return None

    def get_constants(self) -> None:
        return None

    def set_constants(self) -> None:
        return None

    def weight_words(self, sentence: torch.Tensor) -> None:
        # compute weights
        unnormalized_weights: torch.Tensor = torch.ones_like(sentence)
        normalized_weights: torch.Tensor = unnormalized_weights / torch.sum(
            unnormalized_weights
        )

        return normalized_weights


class NaiveBayesWeighter(Weighter):
    def __init__(self) -> None:
        super().__init__()

        # raise NotImplementedError
        return None

    def get_constants(self) -> None:
        return None

    def set_constants(self) -> None:
        return None

    def fit(self) -> None:
        pass

    def weight_words(self, sentence: torch.Tensor) -> None:
        return None


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
        self.embeddings: torch.Tensor = pretrained_model.in_embed

        layers: list = []
        layer_sizes: list = [input_size, *hidden_sizes, output_size]

        # Añadir capas
        for i, _ in enumerate(layer_sizes[:-1]):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.ReLU())

        self.classifier = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, weighter: Weighter) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: input tensor, Dimensions: [batch, channels, height,
                width].

        Returns:
            outputs of the model. Dimensions: [batch, 1].
        """

        embedded_inputs: torch.Tensor = self.embeddings(inputs)
        weights: torch.Tensor = weighter.weight_words(inputs)
        weighted_embedded_inputs: torch.Tensor = torch.einsum(
            "ij,ijk->ik", weights, embedded_inputs
        )
        outputs: torch.Tensor = self.classifier(weighted_embedded_inputs)

        return outputs
