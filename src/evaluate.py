# deep learning libraries
import torch

# from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
# from tqdm.auto import tqdm
from typing import Final
from src.treebank import Tree
from torch.utils.data import DataLoader

from src.data import generate_dataloaders
from src.train_functions import test_step
from src.models import Weighter, UniformWeighter

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device: str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """
    This function is the main program
    """

    batch_size: int = 128

    # load data
    test_loader: DataLoader
    vocab_to_int: dict[str, int]
    test_data: list[Tree]
    (
        _,
        _,
        test_loader,
        vocab_to_int,
        _,
        _,
        _,
        test_data,
    ) = generate_dataloaders(batch_size=batch_size)

    # define name and writer
    name: str = "model_20240422_183617"

    # Load the model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}/{name}.pt").to(device)

    # # Load the constants
    # with open(f"models/{name}/constants.pkl", "rb") as f:
    #     constants = pickle.load(f)
    # model.weighter.set_constants(constants)

    # call test step and evaluate accuracy
    weighter: Weighter = UniformWeighter
    accuracy: float = test_step(model, test_data, device, weighter)
    print(f"accuracy: {accuracy}")

    return None


if __name__ == "__main__":
    main()
