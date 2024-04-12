from src.RecursiveModel.data import download_data, load_trees, load_vocab
from src.RecursiveModel.recursive import RNTN
from src.RecursiveModel.train_functions import train, val, test
from src.RecursiveModel.treebank import Tree
from src.RecursiveModel.utils import get_batch, flatten, save_model

from typing import List, Dict
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR


def main() -> None:
    """
    Args:
    -None

    Returns:
    -None
    """

    download_data()

    # hyperparameters
    hidden_size: int = 300
    lr: float = 0.001
    epochs: int = 50
    batch_size: int = 4
    output_size: int = 5
    step_size: int = 20
    gamma: float = 0.1
    simple_RNN = True

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # empty nohup file
    open("nohup.out", "w").close()

    # Load training, validation and test data
    train_data: List[Tree] = load_trees("train")
    test_data: List[Tree] = load_trees("test")
    val_data: List[Tree] = load_trees("dev")

    word2index: Dict[str, int]
    word2index, _ = load_vocab(train_data)

    # define name and writer
    name: str = (
        f"model_lr_{lr}_hs_{hidden_size}_bs_{batch_size}_e_{epochs}_ss_{step_size}_g{gamma}_simple_{simple_RNN}"
    )
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    with open(f"{name}.txt", "w") as file:
        file.write("GRADIENTES")

    print(device)

    # Define model
    model: RNTN = RNTN(
        word2index=word2index,
        hidden_size=hidden_size,
        output_size=output_size,
        simple_RNN=simple_RNN,
        device=device,
    ).to(device)

    # Define loss function and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler: StepLR = StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in tqdm(range(epochs)):
        train(
            model,
            batch_size,
            train_data,
            device,
            optimizer,
            loss,
            writer,
            epoch,
            name=name,
        )
        val(model, batch_size, val_data, device, loss, writer, epoch)
        # Update the scheduler
        scheduler.step()

    # Save model
    save_model(model, "best_model")

    # accuracy_value: float = test(model, batch_size, test_data, device)
    # print("Test accuracy:", accuracy_value)


if __name__ == "__main__":
    main()
