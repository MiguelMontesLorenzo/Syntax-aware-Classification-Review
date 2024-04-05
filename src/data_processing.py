import os
from nltk.corpus import treebank
import random
import nltk


def download_nltk_data():
    try:
        nltk.data.find("corpora/treebank")
    except LookupError:
        nltk.download("treebank")


def load_ptb_data():
    download_nltk_data()
    sentences = treebank.sents()
    # random.shuffle(sentences)
    return [" ".join(sent) for sent in sentences]


def write_sentences_to_file(sentences, filename):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    with open(filepath, "w") as f:
        for sent in sentences:
            f.write(sent + "\n")


if __name__ == "__main__":
    sentences = load_ptb_data()
    write_sentences_to_file(sentences, "ptb_sentences.txt")
