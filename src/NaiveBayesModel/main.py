import os
import torch

from src.data import download_data
from src.NaiveBayesModel.data_processing import (
    load_sentences,
    build_vocab,
    bag_of_words,
)
from src.NaiveBayesModel.utils import evaluate_classification
from src.NaiveBayesModel.naive_bayes import NaiveBayes
from src.NaiveBayesModel.serial_naive_bayes import SerialNaiveBayes


# modality = "VECTOR-WISE"
modality = "SERIAL"

trn_sample_size = 9645
tst_sample_size = 2210

save_dir = "models"
load_model = False
save_model = False


def main() -> None:
    # Load training data
    trn_path: str
    val_path: str
    tst_path: str
    trn_path, val_path, tst_path = download_data()  # type: ignore

    # Load training data
    trn_path: str
    val_path: str
    tst_path: str
    trn_path, val_path, tst_path = download_data()  # type: ignore

    trn_sentences: list[str]
    trn_labels: list[int]
    val_sentences: list[str]
    val_labels: list[int]
    tst_sentences: list[str]
    tst_labels: list[int]
    sentences: list[str]

    # Load data
    trn_sentences, trn_labels = load_sentences(trn_path)
    val_sentences, val_labels = load_sentences(val_path)
    tst_sentences, tst_labels = load_sentences(tst_path)
    trn_sentences.extend(val_sentences)
    trn_labels.extend(val_labels)

    # Data reduction
    trn_sentences = trn_sentences[:trn_sample_size]
    tst_sentences = tst_sentences[:tst_sample_size]
    trn_labels = trn_labels[:trn_sample_size]
    tst_labels = tst_labels[:tst_sample_size]

    # Build vocabulary
    sentences = trn_sentences + tst_sentences

    print("Building vocabulary...")
    wrd2idx: dict[str, int]
    wrd2idx, _ = build_vocab(sentences)

    # Prepare features and labels for the models
    print("Preparing training BoW...")
    print(f"Total sentences: {len(trn_sentences)}")
    processed_trn_features: list[torch.Tensor] = [
        bag_of_words(sentence, wrd2idx) for sentence in trn_sentences
    ]

    # Prepare features and labels for the models
    print("Preparing testing BoW...")
    print(f"Total sentences: {len(tst_sentences)}")
    processed_tst_features: list[torch.Tensor] = [
        bag_of_words(sentence, wrd2idx) for sentence in tst_sentences
    ]  # type: ignore

    if modality == "VECTOR-WISE":
        # Convert the list of features to a tensor
        trn_features = torch.stack(processed_trn_features)
        trn_labels = torch.tensor(trn_labels, dtype=torch.int)
        tst_features = torch.stack(processed_tst_features)
        tst_labels = torch.tensor(tst_labels, dtype=torch.int)

        print("Training Naive Bayes model...")
        nb_model = NaiveBayes()
        nb_model.fit(trn_features, trn_labels)  # type: ignore

        # Evaluate Naive Bayes model
        print("Evaluating Naive Bayes model...")
        nb_predictions: list[int] = [nb_model.predict(ex) for ex in tst_features]  # type: ignore
        nb_metrics: torch.Dict[str, float] = evaluate_classification(
            torch.tensor(nb_predictions), tst_labels
        )  # type: ignore
        print("Naive Bayes Metrics:", nb_metrics)

    if modality == "SERIAL":
        # Convert the list of features to a tensor
        trn_labels = torch.tensor(trn_labels, dtype=torch.int)  # type: ignore
        tst_labels = torch.tensor(tst_labels, dtype=torch.int)  # type: ignore

        # creating model object
        number_of_classes: int = 5
        nb_model = SerialNaiveBayes(wrd2idx, number_of_classes)  # type: ignore

        print("Training Naive Bayes model...")
        if load_model:
            ckpt_name = os.path.join("stanford_dataset_train")
            ckpt_path = os.path.join(save_dir, "serial", ckpt_name)
            nb_model.load(ckpt_path)  # type: ignore
        else:
            nb_model.fit(torch.tensor(processed_trn_features), trn_labels)  # type: ignore

        # Evaluate Naive Bayes model
        print("Evaluating Naive Bayes model...")
        nb_predictions = [nb_model.predict(ex) for ex in processed_tst_features]  # type: ignore
        nb_metrics = evaluate_classification(
            torch.tensor(nb_predictions), torch.tensor(tst_labels)
        )
        print("Naive Bayes Metrics:", nb_metrics)

        if save_model:
            print("Saving Model ...")
            nb_model.save(save_dir)  # type: ignore


if __name__ == "__main__":
    main()
