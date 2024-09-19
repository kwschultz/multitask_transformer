"""
Direct instantiation of pretrained SentenceTransformer model using the
 sentence-transformers package, extending with PyTorch.
"""
import torch
from sentence_transformers import SentenceTransformer
from typing import List

# Use apple integrated GPU if on macbook
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DualClassifier(torch.nn.Module):
    def __init__(self):
        """
        Load the pretrained SentenceTransformer 'all-MiniLM-L6-v2' as the backbone.
        Create two prediction heads on top of the encoder layers, one for text
        classification and one or sentiment analysis. Randomize weights/bias.
        Max input size: 512 tokens.
        """
        # Load the pretrained MiniLM sentence transformer
        super(DualClassifier, self).__init__()
        self.backbone = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embed_dim = 384

        # MLP for text classification (e.g. classify news articles by type)
        self.num_clf_labels = 6
        self.clf_id_to_label = {
            0: 'sports', 1: 'health', 2: 'tech', 3: 'finance', 4: 'education', 5: 'other'
        }
        self.text_clf = FeedForwardMLP(
            input_dim=self.embed_dim, output_dim=self.num_clf_labels, hidden_dim=192
        )

        # MLP for sentiment analysis (e.g. classify news articles by tone)
        self.num_sentiment_labels = 3
        self.sentiment_id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.sentiment_clf = FeedForwardMLP(
            input_dim=self.embed_dim, output_dim=self.num_sentiment_labels, hidden_dim=192
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass takes sentence embeddings and passes to classifier heads
        :param x: Tensor of sentence embeddings, (384, N)
        :return: Tuple of Tensors, one for each classifier output (384, num_samples)
        """
        # Text Classifier
        text_clf_output = self.text_clf.forward(x)
        # Sentiment analysis
        sentiment_output = self.sentiment_clf.forward(x)

        return text_clf_output, sentiment_output

    def predict(self, sentences: List[str]) -> tuple[List[str], List[str]]:
        """
        Compute the embeddings for input sentences and pass to prediction heads.
        :param sentences: List of strings of text input
        :return: 2 Tensors of shape (384, num_samples), one for text
          classification and one for sentiment
        """
        # Compute embeddings with SentenceTransformer
        sentence_embeddings = self.backbone.encode(
            sentences, convert_to_tensor=True, output_value='sentence_embedding'
        )

        clf_out, sent_out = self.forward(sentence_embeddings)

        text_classes = [self.clf_id_to_label[idx] for idx in clf_out.tolist()]
        sentiments = [self.sentiment_id_to_label[idx] for idx in sent_out.tolist()]

        return text_classes, sentiments


class FeedForwardMLP(torch.nn.Module):
    """
    Simple Multi-Layer Perceptron for Classification heads.
    Fully connected hidden layer, GELU activation and fully connected final
      layer to output class logits.
    """
    def __init__(self, input_dim: int = 384, output_dim: int = 2, hidden_dim: int = 192):
        super(FeedForwardMLP, self).__init__()
        # Layer dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # Layer types
        self.hidden = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.gelu = torch.nn.GELU()
        self.final = torch.nn.Linear(self.hidden_dim, self.output_dim)
        # Random initialization
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        self.hidden.bias.data.fill_(0.02)
        torch.nn.init.xavier_uniform_(self.final.weight)
        self.final.bias.data.fill_(0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply each layer sequentially
        x = self.hidden(x)
        x = self.gelu(x)  # Same activation as embedding layers
        logits = self.final(x)
        pred_classes = torch.argmax(logits, dim=1)
        return pred_classes
