import os
import torch
from transformers import BertModel, BertTokenizer


def load_bert_model():
    """
    Loads basic BERT model and tokenizer. Puts model in evaluation mode.
    """
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.eval()
    return model, tokenizer


def create_embedding(sentence, tokenizer, model):
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    with torch.no_grad():
        outputs = model(torch.tensor([token_ids]))
    embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()
    return embedding


def generate_embeddings(sentences, model, tokenizer, save=False, filename=None):
    embeddings = []

    if save:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)

        with open(filepath, "w") as file:
            for sentence in sentences:
                embedding = create_embedding(sentence, tokenizer, model)
                file.write(" ".join(map(str, embedding)) + "\n")
                embeddings.append(embedding)
    else:
        for sentence in sentences:
            embedding = create_embedding(sentence, tokenizer, model)
            embeddings.append(embedding)

    return embeddings


if __name__ == "__main__":
    model, tokenizer = load_bert_model()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    sentences_file = os.path.join(data_dir, "ptb_sentences.txt")
    with open(sentences_file, "r") as f:
        sentences = [line.strip() for line in f.readlines()]

    embeddings = generate_embeddings(sentences, model, tokenizer, save=True, filename="ptb_embeddings.txt")

