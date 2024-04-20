import os
import torch

from src.data_processing import download_data, load_sentences, build_vocab, bag_of_words
from src.utils import evaluate_classification, list_random_shuffle
from src.naive_bayes import NaiveBayes
from src.serial_naive_bayes import SerialNaiveBayes


modality = "SERIAL"  #"VECTOR-WISE"
trn_sample_size = 20000
tst_sample_size = 20000
save_dir = "models"
load_model = True
save_model = False

def main() -> None:

    # Load training data
    trn_path: str
    val_path: str
    tst_path: str
    trn_path, val_path, tst_path = download_data()

    # Load training data
    trn_path: str
    val_path: str
    tst_path: str
    trn_path, val_path, tst_path = download_data()

    trn_sentences: list[str]
    trn_labels: list[int]
    val_sentences: list[str]
    val_labels: list[int]
    tst_sentences: list[str]
    tst_labels: list[int]
    sentences: list[str]
    labels: list[int]

    # Load data
    trn_sentences, trn_labels = load_sentences(trn_path)
    val_sentences, val_labels = load_sentences(trn_path)
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
    print(f"Total sentences: {len(sentences)}")
    processed_trn_features: list[torch.Tensor] = \
        [bag_of_words(sentence, wrd2idx) for sentence in trn_sentences]

    # Prepare features and labels for the models
    print("Preparing training BoW...")
    print(f"Total sentences: {len(sentences)}")
    processed_tst_features: list[torch.Tensor] = \
        [bag_of_words(sentence, wrd2idx) for sentence in tst_sentences]
    

    if modality == "VECTOR-WISE":

        # Convert the list of features to a tensor
        trn_features = torch.stack(processed_trn_features)
        trn_labels: torch.Tensor = torch.tensor(trn_labels, dtype=torch.int)
        tst_features = torch.stack(processed_tst_features)
        tst_labels: torch.Tensor = torch.tensor(tst_labels, dtype=torch.int)

        print("Training Naive Bayes model...")
        nb_model = NaiveBayes()
        nb_model.fit(trn_features, trn_labels)

        # Evaluate Naive Bayes model
        print("Evaluating Naive Bayes model...")
        nb_predictions: list[int] = [nb_model.predict(ex) for ex in tst_features]
        nb_metrics: torch.Dict[str, float] = evaluate_classification(
            torch.tensor(nb_predictions), tst_labels
        )
        print("Naive Bayes Metrics:", nb_metrics)


    if modality == "SERIAL":

        # Convert the list of features to a tensor
        trn_labels: torch.Tensor = torch.tensor(trn_labels, dtype=torch.int)
        tst_labels: torch.Tensor = torch.tensor(tst_labels, dtype=torch.int)

        # creating model object
        number_of_classes = 5
        nb_model = SerialNaiveBayes(wrd2idx, number_of_classes)

        print("Training Naive Bayes model...")
        if load_model:
            ckpt_name = "20240420_195822"
            ckpt_path = os.path.join(save_dir, ckpt_name)
            nb_model.load(ckpt_path)
        else:
            nb_model.fit(processed_trn_features, trn_labels)

        # Evaluate Naive Bayes model
        print("Evaluating Naive Bayes model...")
        nb_predictions: list[int] = [nb_model.predict(ex) for ex in processed_tst_features]
        nb_metrics: torch.Dict[str, float] = evaluate_classification(
            torch.tensor(nb_predictions), tst_labels
        )
        print("Naive Bayes Metrics:", nb_metrics)

        if save_model:
            print("Saving Model ...")
            nb_model.save(save_dir)


        


if __name__ == "__main__":
    main()
