import torch
from torch.utils.data import DataLoader
import os

from src.Embeddings.data_processing import load_and_preprocess_data, subsample_words, generate_data_loader
from src.Embeddings.skipgram import SkipGramNeg
from src.Embeddings.train import train_skipgram
from src.Embeddings.utils import plot_embeddings, save_model, write_dict

def main():
    print("Starting the SkipGram training pipeline...")

    embedding_dim: int = 300
    batch_size: int = 512
    epochs: int = 5
    learning_rate: float = 0.001
    print_every: int = 1000
    runs_folder: str = "runs"
    model_filename: str = "skipgram_dep_model_updated.pth"
    model_path: str = os.path.join(runs_folder, model_filename)
    train_model: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    csv_file: str = "data/IMDB Dataset.csv"
    json_file: str = "vocab_to_int.json"

    print("Step 1: Loading and preprocessing data...")
    sentences, tokens, correspondences, vocab_to_int, int_to_vocab = load_and_preprocess_data(csv_file)
    print(f"Loaded and preprocessed {len(tokens)} tokens.")
    print(f"Created vocabulary with {len(vocab_to_int)} unique words.")

    if train_model:
        print("Step 3: Subsampling frequent words...")
        train_words, freqs, sampled_correspondences = subsample_words(tokens, vocab_to_int, correspondences)
        print(f"Subsampled words to {len(train_words)} training examples.")

        print("Step 4: Creating DataLoader...")
        dataloader: DataLoader = generate_data_loader(train_words, sentences, sampled_correspondences, batch_size, vocab_to_int, int_to_vocab)

        # Calculate the noise distribution for negative sampling
        print("Calculating noise distribution for negative sampling...")
        word_freqs = torch.tensor(sorted(freqs.values(), reverse=True))
        unigram_dist = word_freqs / word_freqs.sum()
        noise_dist = (unigram_dist ** 0.75 / torch.sum(unigram_dist ** 0.75)).detach().clone().to(device)

        vocab_size: int = len(vocab_to_int)

        print("Step 5: Initializing the SkipGram model...")
        model = SkipGramNeg(vocab_size, embedding_dim, noise_dist=noise_dist).to(device)
        print("Model initialized.")

        print("Step 6: Training the model...")
        train_skipgram(model, dataloader, int_to_vocab, epochs, learning_rate, print_every, device)
        print("Training completed.")

        print("Step 7: Saving the model...")
        save_model(model, model_path)
        print(f"Model saved at {model_path}")

        print("Step 8: Visualizing the word embeddings...")
    else:
        print("Step 3: Loading train model...")
        model: SkipGramNeg = SkipGramNeg(len(vocab_to_int), embedding_dim).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print('Model Loaded.')
        print("Step 4: Visualizing the word embeddings...")

    plot_embeddings(model, int_to_vocab, viz_words=400)
    print("Visualization complete.")

    write_dict(json_file, vocab_to_int)

if __name__ == "__main__":
    main()
