from transformers import AutoConfig, AutoTokenizer
from sentence_transformers import SentenceTransformer


# Load the pretrained MiniLM sentence transformer
PRETRAINED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
config = AutoConfig.from_pretrained(PRETRAINED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
pretrained_model = SentenceTransformer(PRETRAINED_MODEL)

sentences = [
    'The dog ran.',
    'What is the sky?',
    'Who is the president?'
]

if __name__ == '__main__':
    # Given some sentences, compute their embeddings in 384 dim space
    embeddings = pretrained_model.encode(sentences)

    for sent, emb in zip(sentences, embeddings):
        print(f"{sent} \n embedding: {emb[:6]}")


# Do I implement from scratch? Only two hours though.
# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super(FeedForward, self).__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.gelu = nn.GELU()
#
#     def forward(self, x):
#         return self.fc2(self.gelu(self.fc1(x)))
#
# multihead_attn_func = nn.MultiheadAttention(embed_dim, num_heads)
# attn_output, attn_output_weights = multihead_attn_func(query, key, value)

