from typing import List, Dict
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# from torch.optim.lr_scheduler import StepLR

from src.data import download_data, load_trees, generate_dataloaders, load_vocab
from src.utils import save_model, load_pretrained_weights
from src.treebank import Tree
from src.pretrained_embeddings import SkipGramNeg
from src.RecurrentModel.models import RNN
from src.RecurrentModel.train_functions import train_step, val_step, t_step


def main() -> None:
    """
    Args:
    -None

    Returns:
    -None
    """

    download_data()

    # Constants
    NUM_CLASSES: int = 5
    EMBED_DIM: int = 300

    # Hyperparameters
    hidden_size: int = 24
    embedding_dim: int = 300
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 128
    # step_size: int = 20
    # gamma: float = 0.1
    our_embeddings: bool = False

    patience: int = 10
    best_val_loss: float = float("inf")

    # Load training, validation and test data
    train_data: List[Tree] = load_trees("train")

    word2index: Dict[str, int]
    word2index, _ = load_vocab(train_data)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    download_data()

    # empty nohup file
    open("nohup.out", "w").close()

    # define writer
    name: str = "fri_1"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    train_loader, val_loader, test_loader, vocab_to_int, _ = generate_dataloaders(
        batch_size=batch_size
    )

    # our own embeddings
    if our_embeddings:
        pretrained_folder: str = "pretrained"
        pretrained_weights_filename: str = "skipgram_dep_model_updated.pth"
        big_vocab_to_int: str = "vocab_to_int.json"
        vocab_size: int = len(vocab_to_int)

        pretrained_model: SkipGramNeg = SkipGramNeg(vocab_size, embed_dim=EMBED_DIM)
        pretrained_weights_path: str = os.path.join(
            pretrained_folder, pretrained_weights_filename
        )
        pretrained_dict_path: str = os.path.join(pretrained_folder, big_vocab_to_int)

        weights: nn.Embedding = load_pretrained_weights(
            pretrained_weights_path, pretrained_dict_path, vocab_to_int
        )

        pretrained_model.load_pretrained_embeddings(weights)

        # Freeze pretrained model parameters
        for param in pretrained_model.parameters():
            param.requires_grad = False

        # Define model
        model: RNN = RNN(
            word2index=word2index,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_size,
            num_classes=NUM_CLASSES,
            num_layers=1,
            bidirectional=True,
            device=str(device),
            pretrained_model=pretrained_model,
        ).to(device)

    else:
        # Define model
        model = RNN(
            word2index=vocab_to_int,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_size,
            num_classes=NUM_CLASSES,
            num_layers=1,
            bidirectional=True,
            device=str(device),
        ).to(device)

    # Define loss function and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )

    # Learning rate scheduler
    # scheduler: StepLR = StepLR(optimizer, step_size=step_size, gamma=gamma)

    print(f"Training Recurrent model on {device}...")
    print("Len vocab", len(vocab_to_int))

    # Train loop
    for epoch in tqdm(range(epochs)):
        # Call train step
        total_train_lost: float = train_step(
            model, train_loader, loss, optimizer, writer, epoch, device
        )

        # Call val step
        val_step(model, val_loader, loss, writer, epoch, device)

        # Check for Early Stopping
        if total_train_lost < best_val_loss:
            best_val_loss = total_train_lost
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. ")
                print(f"No improvement in test loss for {patience} consecutive epochs.")
                break

        # scheduler.step()

    # Save model
    save_model(model, "best_model")

    accuracy_value: float = t_step(model, test_loader, device)
    print("Test accuracy:", accuracy_value)


if __name__ == "__main__":
    main()
