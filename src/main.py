import torch
import os
try:
    from data_processing import load_and_preprocess_data, create_lookup_tables, subsample_words
    from skipgram import SkipGramNeg
    from train import train_skipgram
    from utils import plot_embeddings, save_model
except:
    from src.data_processing import load_and_preprocess_data, create_lookup_tables, subsample_words
    from src.skipgram import SkipGramNeg
    from src.train import train_skipgram
    from src.utils import plot_embeddings, save_model

def main():
    print("Starting the SkipGram training pipeline...")

    txt_file_path: str = "data/dataset_sentences.txt"
    csv_file_path: str = "data/IMDB Dataset.csv"
    print(f"File path set to {txt_file_path}")
    print(f"File path set to {csv_file_path}")
    
    embedding_dim: int = 300
    batch_size: int = 512
    epochs: int = 5
    learning_rate: float = 0.003
    window_size: int = 5
    print_every: int = 1500
    runs_folder: str = "runs"
    model_filename: str = "skipgram_dep_model.pth"
    model_path: str = os.path.join(runs_folder, model_filename)
    train_model: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Step 1: Loading and preprocessing data...")
    sentences, tokens, correspondences = load_and_preprocess_data(txt_file_path, csv_file_path)
    print(f"Loaded and preprocessed {len(tokens)} tokens.")

    print("Step 2: Creating lookup tables for vocabulary...")
    vocab_to_int, int_to_vocab = create_lookup_tables(tokens)
    print(f"Created vocabulary with {len(vocab_to_int)} unique words.")
    
    if train_model:
        print("Step 3: Subsampling frequent words...")
        train_words, freqs, sampled_correspondences = subsample_words(tokens, vocab_to_int, correspondences)
        print(f"Subsampled words to {len(train_words)} training examples.")

        # Calculate the noise distribution for negative sampling
        print("Calculating noise distribution for negative sampling...")
        word_freqs = torch.tensor(sorted(freqs.values(), reverse=True))
        unigram_dist = word_freqs / word_freqs.sum()
        noise_dist = torch.tensor(unigram_dist ** 0.75 / torch.sum(unigram_dist ** 0.75)).to(device)

        print("Step 4: Initializing the SkipGram model...")
        model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist).to(device)
        print("Model initialized.")

        print("Step 5: Training the model...")
        train_skipgram(model, train_words, sampled_correspondences, sentences, int_to_vocab, vocab_to_int, batch_size, epochs, learning_rate, window_size, print_every, device)
        print("Training completed.")

        print("Step 6: Saving the model...")
        save_model(model, model_path)
        print(f"Model saved at {model_path}")

        print("Step 6: Visualizing the word embeddings...")
    else:
        print("Step 3: Loading train model...")
        model: SkipGramNeg = SkipGramNeg(len(vocab_to_int), embedding_dim).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print('Model Loaded.')
        print("Step 4: Visualizing the word embeddings...")

    plot_embeddings(model, int_to_vocab, viz_words=400)
    print("Visualization complete.")

if __name__ == "__main__":
    main()
