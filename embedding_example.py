"""
Simple testing of the embedding model
"""
from model import DualClassifier
from torch import nn
from itertools import combinations
import pandas as pd


test_text = [
    'The dog ran across the grass.',
    'Why is the sky blue?',
    'The cat jumped into the weeds.',
]


if __name__ == '__main__':
    model = DualClassifier()

    # Given some sentences, compute their embeddings in 384 dim space
    embeddings = model.backbone.encode(test_text, convert_to_tensor=True)
    print(f"Embeddings size: {embeddings.shape}")

    # Compute cosine similarity between each sentence
    cos_sim = nn.CosineSimilarity(dim=0)
    sentence_pairs = combinations(range(len(embeddings)), 2)

    score_data = []

    for i1, i2 in sentence_pairs:
        score = cos_sim(embeddings[i1], embeddings[i2]).item()
        score_data.append({
            'cosine_sim': score, 'sentence_1': test_text[i1], 'sentence_2': test_text[i2]
        })

    score_df = pd.DataFrame(score_data).sort_values('cosine_sim', ascending=False)
    pd.set_option('display.max_columns', 5)
    pd.set_option('display.width', 1000)

    print(score_df)
