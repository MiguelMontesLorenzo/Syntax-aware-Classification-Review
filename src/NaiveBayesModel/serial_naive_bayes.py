import torch
import datetime
import os


class SerialNaiveBayes:
    def __init__(self, vocabulary: list[str], output_size: int = 0) -> None:
        """
        Initializes the Naive Bayes classifier

        Args:
            vocabulary (list[str]): List of unique words in the vocabulary.
            output_size (int): Number of classes in the dataset.

        Returns:
            None
        """

        self.vocabulary: list[str] = vocabulary
        self.vocab_size = len(vocabulary)
        self.output_size = output_size

        self.log_words_tensor = None
        self.log_class_tensor = None

        return None

    def fit(
        self, sentences: list[torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            sentences (list[torch.Tensor]):
                List of bag of words representations of the training examples.
            labels (torch.Tensor):
                Labels corresponding to each training example.
        Returns:
            (tuple[torch.Tensor, torch.Tensor]):
                A tuple of tensors containing the log of the conditional probabilities
                and the log of the class priors.
        """

        if self.output_size == 0:
            unique_labels: torch.Tensor = torch.unique(labels, sorted=True)
            self.output_size: int = unique_labels.shape[0]

        words_tensor = torch.zeros(size=(self.vocab_size, self.output_size))
        class_tensor = torch.zeros(size=(self.output_size,))

        # compute counts
        for i, sentence in enumerate(sentences):
            if i % 1000 == 0:
                print(f"Processing sentence {i}")

            sentence_label = int(labels[i].item())
            class_tensor[sentence_label] += 1
            words_tensor[:, sentence_label] += sentence.to(dtype=torch.float)

        # compute probabilities
        sigma: int = 1
        numerator = words_tensor + sigma
        denominator = (
            torch.sum(words_tensor, axis=0, keepdims=True) + sigma * self.vocab_size
        )
        words_prob = numerator / denominator
        class_prob = class_tensor / torch.sum(class_tensor, axis=0)

        # compute & save log probabilities
        self.log_words_tensor: torch.Tensor = torch.log(words_prob)
        self.log_class_tensor: torch.Tensor = torch.log(class_prob)

        return (words_prob, class_prob)

    def predict_probabilities(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Predicts the class log-probabilities for the given features.

        Args:
            sentence (torch.Tensor):
                Bag of words representation of the sentence example to classify.
        Returns:
            (torch.Tensor):
                A tensor containing the log of the conditional probabilities.
        """

        conditional_log_probs = torch.einsum("ij,i->j", self.log_words_tensor, sentence)
        log_probs = conditional_log_probs + self.log_class_tensor
        return log_probs

    def predict(self, sentence: torch.Tensor) -> int:
        """
        Predicts the class labels for the given features.

        Args:
            sentence (torch.Tensor):
                Bag of words representation of the sentence example to classify.
        Returns:
            (int):
                The predicted class label.
        """

        log_probs = self.predict_probabilities(sentence)
        return torch.argmax(log_probs).item()

    def save(self, path: str) -> None:
        """
        Saves the model to the given path.

        Args:
            path (str):
                Path to save the model.
        Returns:
            None
        """
        time_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = os.path.join(path, "serial", time_string)

        # create directory
        if not os.path.exists(ckpt_name):
            os.makedirs(ckpt_name)

        # save model parameters
        words_path = os.path.join(ckpt_name, f"log_words_tensor.pt")
        class_path = os.path.join(ckpt_name, f"log_class_tensor.pt")

        print(f"Saving model parameters to {ckpt_name}")

        torch.save(self.log_words_tensor, words_path)
        torch.save(self.log_class_tensor, class_path)

        return None

    def load(self, path) -> None:
        """
        Loads the model from the given path.
        Args:
            path (str):
                Path containing the model.
        Returns:
            None
        """

        print(f"Loading model parameters from {path}")

        words_path = os.path.join(path, "serial", "log_words_tensor.pt")
        class_path = os.path.join(path, "serial", "log_class_tensor.pt")

        self.log_words_tensor = torch.load(words_path)
        self.log_class_tensor = torch.load(class_path)

        return None
