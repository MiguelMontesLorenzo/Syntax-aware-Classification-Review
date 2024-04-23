import torch
import os

from src.data import generate_dataloaders
from src.utils import load_pretrained_weights
from src.treebank import Tree
from src.pretrained_embeddings import SkipGramNeg
from src.models import VecAvg, Weighter, UniformWeighter, NaiveBayesWeighter

from torch.utils.data import DataLoader

# from typing import Final
from src.train_functions import test_step

# params
EMBED_DIM: int = 300
NUM_CLASSES: int = 5
HIDDEN_SIZES: tuple[int, int, int] = (264, 264, 64)
use_naive_bayes: bool = True

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def main() -> None:
    """
    This function is the main program
    """

    batch_size: int = 128

    # load data
    test_loader: DataLoader
    vocab_to_int: dict[str, int]
    train_data: list[Tree]
    val_data: list[Tree]
    (_, _, test_loader, vocab_to_int, _, train_data, val_data, _) = (
        generate_dataloaders(batch_size=batch_size)
    )

    pretrained_folder: str = "pretrained"
    pretrained_weights_filename: str = "skipgram_dep_model_updated.pth"
    big_vocab_to_int: str = "vocab_to_int.json"
    vocab_size: int = len(vocab_to_int)

    pretrained_model: SkipGramNeg = SkipGramNeg(vocab_size, embed_dim=EMBED_DIM)
    pretrained_weights_path: str = os.path.join(
        pretrained_folder, pretrained_weights_filename
    )
    pretrained_dict_path: str = os.path.join(pretrained_folder, big_vocab_to_int)

    weights: torch.Tensor = load_pretrained_weights(
        pretrained_weights_path, pretrained_dict_path, vocab_to_int
    )

    pretrained_model.load_pretrained_embeddings(weights)

    # Freeze pretrained model parameters
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # define name and writer
    name: str = "model_name"

    naive_bayes_params: tuple[list[list[int]], list[int]]
    sentences: list[list[int]] = []
    labels: list[int] = []
    weighter: Weighter
    for tree in train_data:
        sentences.append([vocab_to_int[word] for word in tree.get_words()])
        labels.append(tree.labels[-1])
    for tree in val_data:
        sentences.append([vocab_to_int[word] for word in tree.get_words()])
        labels.append(tree.labels[-1])

    unique_labels: torch.Tensor = torch.unique(torch.Tensor(labels))

    if use_naive_bayes:
        naive_bayes_params = (sentences, labels)
        weighter = NaiveBayesWeighter(
            vocab_size=vocab_size, output_size=unique_labels.shape[0]
        )
        weighter.fit(*naive_bayes_params)
    else:
        weighter = UniformWeighter(
            vocab_size=vocab_size, output_size=unique_labels.shape[0]
        )
    # Load the model
    model: torch.nn.Module = VecAvg(
        input_size=EMBED_DIM,
        output_size=NUM_CLASSES,
        hidden_sizes=HIDDEN_SIZES,
        pretrained_model=pretrained_model,
    ).to(device)
    model.load_state_dict(torch.load(f"models/{name}/{name}.pth"))

    # call test step and evaluate accuracy
    weighter: Weighter = UniformWeighter()
    accuracy: float = test_step(model, test_loader, device, weighter, NUM_CLASSES)
    print(f"accuracy: {accuracy}")

    return None


if __name__ == "__main__":
    main()
