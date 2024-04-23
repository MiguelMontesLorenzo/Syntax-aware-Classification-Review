import torch
import os

from datetime import datetime
from tqdm.auto import tqdm

# from torch.optim.lr_scheduler import StepLR

from src.data import download_data, generate_dataloaders
from src.utils import save_model, load_pretrained_weights
from src.treebank import Tree
from src.pretrained_embeddings import SkipGramNeg
from src.models import VecAvg, Weighter, UniformWeighter, NaiveBayesWeighter
from src.train_functions import train_step, val_step

from torch.utils.data import DataLoader

# hyperparameters
EMBED_DIM: int = 300
NUM_CLASSES: int = 5
HIDDEN_SIZES: tuple[int, int, int] = (256, 256, 64)  # (514, 514, 64)
epochs: int = 100
lr: float = 5e-3
batch_size: int = 128
use_naive_bayes: bool = True


def main() -> None:
    """
    This function is the main program for training.
    """
    # empty nohup file
    open("nohup.out", "w").close()

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_data()

    # load data
    train_loader: DataLoader
    val_loader: DataLoader
    vocab_to_int: dict[str, int]
    train_data: list[Tree]
    val_data: list[Tree]
    (
        train_loader,
        val_loader,
        _,
        vocab_to_int,
        _,
        train_data,
        val_data,
        _,
    ) = generate_dataloaders(batch_size=batch_size)

    # if our own embeddings \
    if True:
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

        model: torch.nn.Module = VecAvg(
            input_size=EMBED_DIM,
            output_size=NUM_CLASSES,
            hidden_sizes=HIDDEN_SIZES,
            pretrained_model=pretrained_model,
        ).to(device)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train loop
    for epoch in tqdm(range(epochs)):
        # call train step
        print()
        train_step(
            model, train_loader, loss, optimizer, epoch, device, weighter, NUM_CLASSES
        )
        val_step(model, val_loader, loss, epoch, device, weighter, NUM_CLASSES)

    # save model
    now: datetime = datetime.now()
    current_time: str = now.strftime("%Y%m%d_%H%M%S")
    model_name: str = f"model_{current_time}"
    save_model(model, model_name)

    return None


if __name__ == "__main__":
    main()
