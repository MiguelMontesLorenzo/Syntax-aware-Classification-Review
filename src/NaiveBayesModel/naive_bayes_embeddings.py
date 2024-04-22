import torch
from typing import Dict


class NaiveBayes:
    def __init__(self) -> None:
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = {}
        self.conditional_probabilities: Dict[int, torch.Tensor] = {}
        self.vocab_size: int = 0
        self.labels: torch.Tensor

    def fit(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0
    ) -> None:
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating
        conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training
            examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """

        self.labels = labels
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = labels.shape[0]
        self.conditional_probabilities = self.estimate_conditional_probabilities(
            features, labels, delta
        )

        return None

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their
            estimated prior probabilities.
        """

        unique_integers: torch.Tensor
        counts: torch.Tensor
        unique_integers, counts = torch.unique(labels, return_counts=True)
        collection_size: float = float(labels.shape[0])

        class_priors: Dict[int, torch.Tensor] = {
            int(key): torch.tensor(int(value)) / collection_size
            for key, value in zip(unique_integers.tolist(), counts.tolist())
        }

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace
        smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training
            examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each
            class.
        """

        word_probs_by_class: Dict[int, torch.Tensor] = dict({})

        class_labels: list = labels.unique().tolist()
        smoothed_vocabulary_word_frecuencies: torch.Tensor = (
            torch.sum(features, dim=0) + labels.shape[0] * delta
        )

        for class_label in class_labels:
            # binary vector indicating which examples belong to considered class
            bin: torch.Tensor = torch.where(
                labels == class_label,
                torch.tensor(1, dtype=torch.float),
                torch.tensor(0, dtype=torch.float),
            )
            bin_diag_matrix: torch.Tensor = torch.diag(bin)
            classbags_features: torch.Tensor = bin_diag_matrix @ features

            smoothed_class_word_count: torch.Tensor = (
                torch.sum(classbags_features, dim=0) + delta
            )
            class_cond_prob: torch.Tensor = (
                smoothed_class_word_count / smoothed_vocabulary_word_frecuencies
            )

            word_probs_by_class[class_label] = class_cond_prob

        return word_probs_by_class

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )

        log_posteriors: torch.Tensor = torch.empty(size=self.labels.shape)

        log_priors: dict[int, torch.Tensor] = {
            i: torch.log(class_prior) for i, class_prior in self.class_priors.items()
        }
        log_conds: dict[int, torch.Tensor] = {
            i: torch.log(class_cond_prob)
            for i, class_cond_prob in self.conditional_probabilities.items()
        }

        for i, class_label in enumerate(self.labels.tolist()):
            log_posteriors[class_label] = (
                feature @ log_conds[class_label] + log_priors[class_label]
            )

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> float:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of
            the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)
        pred: float = torch.argmax(log_posteriors).item()

        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of
            the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all
            classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)
        probs: torch.Tensor = torch.softmax(log_posteriors, dim=-1)

        return probs
