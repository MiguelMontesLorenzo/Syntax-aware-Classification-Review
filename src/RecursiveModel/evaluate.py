from src.RecursiveModel.train_functions import test
from typing import List
from src.RecursiveModel.data import load_trees
from src.RecursiveModel.treebank import Tree
from src.RecursiveModel.recursive import RNTN
from src.RecursiveModel.utils import set_seed
import torch


set_seed(42)


def main() -> None:
    """
    Evaluate the best_model
    Args:
    - None
    Returns
    - None
    """
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size: int = 64
    name: str = "best_model"
    test_data: List[Tree] = load_trees("test")
    model: RNTN = torch.load(f"models/{name}.pt", map_location="cpu")
    model = model.to(device)
    accuracy_value: float = test(model, batch_size, test_data, device)
    print("Test accuracy:", accuracy_value)


if __name__ == "__main__":
    main()
