from src_bert.data_processing import load_ptb_data
from src_bert.embeddings import load_bert_model, generate_embeddings


if __name__ == "__main__":
    sentences: list = load_ptb_data()

    model, tokenizer = load_bert_model()
    embeddings: list = generate_embeddings(sentences, model, tokenizer, save=True, filename="ptb_embeddings.txt")

    print("Embeddings generated and saved successfully!")
