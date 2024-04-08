import torch
import torch.optim as optim
from typing import List, Dict

try:
    from src.skipgram import SkipGramNeg, NegativeSamplingLoss  
    from src.data_processing import get_batches, cosine_similarity
except ImportError:
    from skipgram import SkipGramNeg, NegativeSamplingLoss
    from data_processing import get_batches, cosine_similarity

def train_skipgram(model: SkipGramNeg,
                   words: List[int],
                   sampled_correspondences: Dict[int, str],
                   sentences: List[str],
                   int_to_vocab: Dict[int, str],
                   vocab_to_int: Dict[str, int], 
                   batch_size: int = 512, 
                   epochs: int = 5,
                   learning_rate: float = 0.003, 
                   window_size: int = 5, 
                   print_every: int = 1500,
                   device: str = "cpu"):
    """
    Trains the SkipGram model using negative sampling.

    Args:
    - model (SkipGram): SkipGram model to be trained.
    - words (List[int]): list of words (integers) to train on.
    - sampled_correspondences (Dict[int, str]): association of word index in tokens and the index of the sentence that
    word belongs to.
    - sentences (List[str]): list of correctly processed sentences fomr both files.
    - int_to_vocab (Dict[int, str]): dictionary mapping integers back to vocabulary words.
    - batch_size (int): size of each batch of input and target words.
    - epochs (int): The number of epochs to train for.
    - learning_rate (float): learning rate for the optimizer.
    - window_size (int): the size of the context window for generating training pairs.
    - print_every (int): the frequency of printing the training loss and validation examples.
    - device (str): the device (CPU or GPU) where the tensors will be allocated.
    """
    # Define loss and optimizer
    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    steps: int = 0
    n_samples: int = 3
    # Training loop
    for epoch in range(epochs):
        
        for input_words, target_words in get_batches(words, sampled_correspondences, sentences, batch_size, vocab_to_int, window_size):
            steps += 1
            # Convert inputs and context words into tensors
            inputs = torch.LongTensor(input_words)
            targets = torch.LongTensor(target_words)
            inputs, targets = inputs.to(device), targets.to(device)

            # input, output, and noise vectors
            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_output(targets)
            noise_vectors = model.forward_noise(input_vectors.size(0), n_samples)
            
            # negative sampling loss
            loss = criterion(input_vectors, output_vectors, noise_vectors)

            # Backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if steps % print_every == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Step: {steps}, Loss: {loss.item()}")
                # Cosine similarity
                valid_examples, valid_similarities = cosine_similarity(model.in_embed, device=device)
                _, closest_idxs = valid_similarities.topk(6)

                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
                print("...\n")
