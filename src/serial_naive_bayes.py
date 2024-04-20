import torch
from typing import Dict



class SerialNaiveBayes:
    def __init__(self, vocabulary, output_size=0) -> None:
        """
        Initializes the Naive Bayes classifier
        """
        # self.class_priors: Dict[int, torch.Tensor] = None
        # self.conditional_probabilities: Dict[int, torch.Tensor] = None
        # self.vocab_size: int = None
        # self.labels: torch.Tensor

        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.output_size = output_size

        self.log_words_tensor = None
        self.log_class_tensor = None

    def fit(
            self, 
            sentences: list[torch.Tensor], 
            labels: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        """

        if self.output_size == 0:
            unique_labels = torch.unique(labels, sorted=True)
            self.output_size = unique_labels.shape[0]

        words_tensor = torch.zeros(size=(self.vocab_size, self.output_size))
        class_tensor = torch.zeros(size=(self.output_size,))

        # compute counts
        for i, sentence in enumerate(sentences):
            
            if i % 1000 == 0:
                print(f"Processing sentence {i}")

            sentence_label = int(labels[i].item() - 1)
            class_tensor[sentence_label] += 1
            for word_idx in sentence:
                idx = int(word_idx.item())
                words_tensor[idx][sentence_label] += 1

        
        # compute probabilities
        sigma = 1
        numerator = words_tensor + sigma
        denominator = torch.sum(words_tensor, axis=0, keepdims=True) + \
            sigma * self.vocab_size
        words_prob = numerator / denominator
        class_prob = class_tensor / torch.sum(class_tensor, axis=0)

        # compute & save log probabilities
        self.log_words_tensor = torch.log(words_prob)
        self.log_class_tensor = torch.log(class_prob)

        return (words_prob, class_prob)
    
    def predict_probabilities(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Predicts the class labels for the given features.
        """

        conditional_log_probs = torch.einsum('ij,i->j', self.log_words_tensor, sentence)
        log_probs = conditional_log_probs + self.log_class_tensor
        probs = torch.exp(log_probs)

        return probs
    
    def predict(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Predicts the class labels for the given features.
        """

        probs = self.predict_probabilities(sentence)
        return torch.argmax(probs)