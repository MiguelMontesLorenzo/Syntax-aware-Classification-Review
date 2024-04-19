from typing import List, Dict
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
import os

from src.data import download_data, load_trees, generate_dataloaders
from src.treebank import Tree
from src.utils import load_pretrained_weights, save_model
from src.pretrained_embeddings import SkipGramNeg
from src.RecursiveModel.data import load_vocab
from src.RecursiveModel.recursive import RNTN
from src.RecursiveModel.train_functions import train, val
from src.RecursiveModel.utils import set_seed

set_seed(42)


def main(start_epoch: int = 0) -> None:
    """
    Args:
    -None

    Returns:
    -None
    """

    download_data()

    # hyperparameters
    hidden_size: int = 300
    lr: float = 0.02
    epochs: int = 1
    batch_size: int = 4
    output_size: int = 5
    step_size: int = 20
    gamma: float = 0.1
    simple_RNN: bool = True
    our_embeddings = True
    patience: int = 20
    best_val_loss: float = float("inf")

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # empty nohup file
    open("nohup.out", "w").close()

    # Load training, validation and test data
    train_data: List[Tree] = load_trees("train")
    # test_data: List[Tree] = load_trees("test")
    val_data: List[Tree] = load_trees("dev")

    word2index: Dict[str, int]
    word2index, _ = load_vocab(train_data)

    # define name and writer
    name: str = (
        f"model_lr_{lr}_hs_{hidden_size}_bs_{batch_size}_e_{epochs}_ss_{step_size}_g{gamma}_simple_{simple_RNN}"
    )
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    print(device)

    # our own embeddings
    if our_embeddings:
        pretrained_folder: str = "pretrained"
        pretrained_weights_filename: str = "skipgram_dep_model_updated.pth"
        big_vocab_to_int: str = "vocab_to_int.json"

        _, _, _, vocab_to_int, _ = generate_dataloaders(batch_size=batch_size)

        vocab_size: int = len(vocab_to_int)

        pretrained_model: SkipGramNeg = SkipGramNeg(vocab_size, embed_dim=hidden_size)
        pretrained_weights_path: str = os.path.join(pretrained_folder, pretrained_weights_filename)
        pretrained_dict_path: str = os.path.join(pretrained_folder, big_vocab_to_int)

        weights: torch.Tensor = load_pretrained_weights(pretrained_weights_path, pretrained_dict_path, vocab_to_int)

        pretrained_model.load_pretrained_embeddings(weights)
        
        # Define model
        model: RNTN = RNTN(
            word2index=word2index,
            hidden_size=hidden_size,
            output_size=output_size,
            simple_RNN=simple_RNN,
            device=device,
            pretrained_model=pretrained_model
        ).to(device)

    else:
        # Define model
        model: RNTN = RNTN(
            word2index=word2index,
            hidden_size=hidden_size,
            output_size=output_size,
            simple_RNN=simple_RNN,
            device=device
        ).to(device)

    if start_epoch > 0:
        checkpoint_path = f"models/checkpoint_{start_epoch}"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            print(
                f"Checkpoint file '{checkpoint_path}' not found. Training from scratch."
            )

    # Define loss function and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler: StepLR = StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs):
        train(
            model,
            batch_size,
            train_data,
            device,
            optimizer,
            loss,
            writer,
            epoch,
        )
        val_loss: float = val(model, batch_size, val_data, device, loss, writer, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"No improvement for {patience} epochs. Early stopping...")
                break

        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            if not os.path.isdir("models"):
                os.makedirs("models")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                f"models/checkpoint_{epoch+1}",
            )

        # Update the scheduler
        scheduler.step()

    # Save model
    save_model(model, "best_model")


if __name__ == "__main__":
    main(5)
