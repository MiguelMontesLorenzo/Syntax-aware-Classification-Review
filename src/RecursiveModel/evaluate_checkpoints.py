from typing import List, Dict
import torch
import os
 
from src.data import download_data, load_trees, generate_dataloaders, load_vocab
from src.treebank import Tree
from src.utils import load_pretrained_weights, save_model
from src.pretrained_embeddings import SkipGramNeg
from src.RecursiveModel.recursive import RNTN
from src.RecursiveModel.train_functions import train, test
from src.RecursiveModel.utils import set_seed
 
set_seed(42)
 
 
def main(num: int) -> None:
    """
    Evaluate the best_model
    Args:
    - None
    Returns
    - None
    """
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    hidden_size: int = 300
    lr: float = 0.02
    epochs: int = 30
    batch_size: int = 4
    output_size: int = 5
    step_size: int = 20
    gamma: float = 0.1
    simple_RNN: bool = False
    our_embeddings = True
    patience: int = 20
    best_val_loss: float = float("inf")
 
    # empty nohup file
    open("nohup.out", "w").close()
 
    # Load training, validation and test data
    train_data: List[Tree] = load_trees("train")
 
    word2index: Dict[str, int]
    word2index, _ = load_vocab(train_data)
 
    print(device)
 
    name: str = f"checkpoint_{num}"
 
    # our own embeddings
    if our_embeddings:
        pretrained_folder: str = "pretrained"
        pretrained_weights_filename: str = "skipgram_dep_model_updated.pth"
        big_vocab_to_int: str = "vocab_to_int.json"

        _, _, _, vocab_to_int, _ = generate_dataloaders(batch_size=batch_size)

        vocab_size: int = len(vocab_to_int)

        pretrained_model: SkipGramNeg = SkipGramNeg(vocab_size, embed_dim=hidden_size)
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

        # Define model
        model: RNTN = RNTN(
            word2index=word2index,
            hidden_size=hidden_size,
            output_size=output_size,
            simple_RNN=simple_RNN,
            device=device,
            pretrained_model=pretrained_model,
        ).to(device)

    else:
        # Define model
        model: RNTN = RNTN(
            word2index=word2index,
            hidden_size=hidden_size,
            output_size=output_size,
            simple_RNN=simple_RNN,
            device=device,
        ).to(device)

 
    model.load_state_dict(torch.load(f"models/{name}")["model_state_dict"])
 
    test_data: List[Tree] = load_trees("test")

    accuracy_value: float = test(model, batch_size, test_data, device)
    print("Test accuracy:", accuracy_value)
 
 
if __name__ == "__main__":
    main(10)