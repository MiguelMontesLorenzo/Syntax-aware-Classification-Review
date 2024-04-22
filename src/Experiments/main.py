from typing import List

import torch
import matplotlib.pyplot as plt

from src.data import download_data, load_trees
from src.treebank import Tree
from src.RecursiveModel.recursive import RNTN
from src.RecursiveModel.utils import set_seed
from src.Experiments.utils import read_indices, count_correct


set_seed(42)


def main() -> None:
    """
    Args:
    -None

    Returns:
    -None
    """

    download_data()
    train_data: List[Tree] = load_trees("train")

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_recursive: str = "RecNN_55.26"
    name_RTNT: str = "RTNT_58.19"
    name_recursive_2: str = "RecNN_63.03_embeddings_propios"
    name_RTNT_2: str = "RNTN_62.52_embeddings_propios"

    # Load all models models
    model_simple: RNTN = torch.load(
        f"models/{name_recursive}.pt", map_location=torch.device("cpu")
    )
    model_simple.device = device
    model_simple.to(device)

    ##
    model_simple_2: RNTN = torch.load(
        f"models/{name_recursive_2}.pt", map_location=torch.device("cpu")
    )
    model_simple_2.device = device
    model_simple_2.to(device)

    ##
    model_RTNT: RNTN = torch.load(
        f"models/{name_RTNT}.pt", map_location=torch.device("cpu")
    )
    model_RTNT.device = device
    model_RTNT.to(device)

    model_RTNT_2: RNTN = torch.load(
        f"models/{name_RTNT_2}.pt", map_location=torch.device("cpu")
    )
    model_RTNT_2.device = device
    model_RTNT_2.to(device)

    # Get the trees of the negated phrases
    filepath: str = "./src/Experiments/negation_phrases_indexes.txt"
    indices: List[int] = read_indices(filepath=filepath)
    filtered_trees: List[Tree] = [train_data[indice] for indice in indices]

    # Get real labels
    labels: List[int] = []
    for indice in indices:
        label: int = train_data[indice].labels[-1]
        labels.append(label)

    # Compute outputs
    outputs_simple: torch.Tensor = model_simple(filtered_trees, root_only=True)
    real_predictions_simple: torch.Tensor = torch.argmax(outputs_simple, dim=1)
    correct_predictions_simple: int = count_correct(real_predictions_simple, labels)

    outputs_simple_2: torch.Tensor = model_simple_2(filtered_trees, root_only=True)
    real_predictions_simple_2: torch.Tensor = torch.argmax(outputs_simple_2, dim=1)
    correct_predictions_simple_2: int = count_correct(real_predictions_simple_2, labels)

    outputs_RTNT: torch.Tensor = model_RTNT(filtered_trees, root_only=True)
    real_predictions_RTNT: torch.Tensor = torch.argmax(outputs_RTNT, dim=1)
    correct_predictions_RTNT: int = count_correct(real_predictions_RTNT, labels)

    outputs_RTNT_2: torch.Tensor = model_RTNT_2(filtered_trees, root_only=True)
    real_predictions_RTNT_2: torch.Tensor = torch.argmax(outputs_RTNT_2, dim=1)
    correct_predictions_RTNT_2: int = count_correct(real_predictions_RTNT_2, labels)

    print(labels)
    print(
        real_predictions_simple,
        real_predictions_simple_2,
        real_predictions_RTNT,
        real_predictions_RTNT_2,
    )
    # Plot results
    models: List[str] = [
        "Simple_torch_embeddings",
        "Simple_own_embeddings",
        "RNTN_torch_embeddings",
        "RNTN_own_embeddings",
    ]
    counts: List[int] = [
        correct_predictions_simple,
        correct_predictions_simple_2,
        correct_predictions_RTNT,
        correct_predictions_RTNT_2,
    ]

    plt.bar(models, counts, color="maroon", width=0.4)

    plt.xlabel("Models")
    plt.ylabel("Correct predictions")
    plt.title("Correct predictions of negated sentences")
    plt.show()


if __name__ == "__main__":
    main()
