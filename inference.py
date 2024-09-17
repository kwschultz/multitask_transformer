import torch
from transformers import AutoTokenizer
import torch.nn.functional as F


# Adapted from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


# Model input
sentences = ['The dog ran across the grass.', 'What color is the sky?']

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)

