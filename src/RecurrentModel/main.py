import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
try:
    from src.RecurrentModel.data import download_data, generate_dataloaders
    from src.RecurrentModel.utils import save_model, plot_embeddings
    from src.RecurrentModel.pretrained_embeddings import SkipGramNeg
    from src.RecurrentModel.models import RNN
    from src.RecurrentModel.train_functions import train_step, val_step, t_step
except ImportError:
    from data import download_data, generate_dataloaders
    from utils import save_model, plot_embeddings
    from pretrained_embeddings import SkipGramNeg
    from models import RNN
    from train_functions import train_step, val_step, t_step


def main() -> None:
    """
    Args:
    -None

    Returns:
    -None
    """
    # Constants
    NUM_CLASSES: int = 5
    EMBED_DIM: int = 300

    # Hyperparameters
    hidden_size: int = 300
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 128
    step_size: int = 20
    gamma: float = 0.1

    patience: int = 10
    best_val_loss: float = float("inf")

    pretrained_folder: str = "pretrained"
    pretrained_model_filename: str = "skipgram_dep_model.pth"

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_data()

    # empty nohup file
    open("nohup.out", "w").close()

    train_loader, val_loader, test_loader, vocab_to_int, int_to_vocab = generate_dataloaders(batch_size=batch_size)

    vocab_size: int = len(vocab_to_int)

    pretrained_model: SkipGramNeg = SkipGramNeg(vocab_size, embed_dim=EMBED_DIM)
    model_path: str = os.path.join(pretrained_folder, pretrained_model_filename)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    pretrained_model.load_pretrained_embeddings(state_dict["in_embed.weight"])

    # plot_embeddings(pretrained_model, int_to_vocab, viz_words=400)

    # Freeze pretrained model parameters
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # define name and writer
    # name: str = (
    #     f"model_lr_{learning_rate}_hs_{hidden_size}_bs_{batch_size}_e_{epochs}_ss_{step_size}_g{gamma}"
    # )
    name: str = "tue_1"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")
    
    # Define model
    model: RNN = RNN(
        pretrained_model=pretrained_model,
        hidden_dim=hidden_size,
        num_classes=NUM_CLASSES,
        num_layers=1,
        bidirectional=True,
    ).to(device)

    # Define loss function and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler: StepLR = StepLR(optimizer, step_size=step_size, gamma=gamma)

    print(f"Training Recurrent model on {device}...")

    # Train loop
    for epoch in tqdm(range(epochs)):
        # Call train step
        total_train_lost: float = train_step(model, train_loader, loss, optimizer, writer, epoch, device)

        # Call val step
        val_step(model, val_loader, loss, writer, epoch, device)

        # Check for Early Stopping
        if total_train_lost < best_val_loss:
            best_val_loss = total_train_lost
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. No improvement in test loss for {patience} consecutive epochs.")
                break
        
        scheduler.step()

    # Save model
    save_model(model, "best_model")

    accuracy_value: float = t_step(model, test_loader, device)
    print("Test accuracy:", accuracy_value)


if __name__ == "__main__":
    main()