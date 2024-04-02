import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

sentence = "The quick brown fox jumps over the lazy dog."

tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

with torch.no_grad():
    outputs = bert_model(torch.tensor([token_ids]))

embeddings = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
attention_mask = outputs.attentions[-1].squeeze(0)  # attention mask for the last layer

dependency_tree = {
    "root": {
        "word": "jumps",
        "children": [
            {"word": "fox", "relation": "subj"},
            {"word": "dog", "relation": "obj"},
            {"word": "quick", "relation": "amod"},
            {"word": "brown", "relation": "amod"},
            {"word": "over", "relation": "prep"},
            {"word": "the", "relation": "pobj"},
            {"word": "lazy", "relation": "amod"},
            {"word": "The", "relation": "det"}
        ]
    }
}

def modify_embeddings(embeddings, dependency_tree):
    modified_embeddings = embeddings.clone()

    for node in dependency_tree.values():
        word = node["word"]
        children = node.get("children", [])
        if children:
            children_embeddings = torch.stack([modified_embeddings[:, tokenizer.convert_tokens_to_ids(child["word"])] for child in children], dim=1)
            mean_children_embeddings = torch.mean(children_embeddings, dim=1, keepdim=True)
            word_embedding = modified_embeddings[:, tokenizer.convert_tokens_to_ids(word)]
            modified_embeddings[:, tokenizer.convert_tokens_to_ids(word)] = word_embedding + mean_children_embeddings

    return modified_embeddings

modified_embeddings = modify_embeddings(embeddings, dependency_tree)
