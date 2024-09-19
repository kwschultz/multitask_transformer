"""
Direct instantiation of pretrained SentenceTransformer model using the
 sentence-transformers package, extending with PyTorch.
"""
import torch
from sentence_transformers import SentenceTransformer
from typing import List


device = "cuda" if torch.cuda.is_available() else "cpu"


class DualClassifier(SentenceTransformer):
    def __init__(self):
        """
        Load the pretrained SentenceTransformer 'all-MiniLM-L6-v2' as the backbone.
        Create two prediction heads on top of the encoder layers, one for text
        classification and one or sentiment analysis. Randomize weights/bias.
        Max input size: 512 tokens.
        """
        # Load the pretrained MiniLM sentence transformer
        super(DualClassifier, self).__init__('sentence-transformers/all-MiniLM-L6-v2')
        self.embed_dim = 384
        self.max_input_tokens = 512

        # 2 layer MLP for text classification (e.g. classify news articles by type)
        self.num_clf_labels = 6
        self.clf_id_to_label = {0: 'sports', 1: 'health', 2: 'tech', 3: 'finance', 4: 'education', 5: 'other'}
        self.text_clf = FeedForwardMLP(
            input_dim=self.embed_dim, output_dim=self.num_clf_labels, hidden_dim=192
        )

        # 2 layer MLP for sentiment analysis (e.g. classify news articles by tone)
        self.num_sentiment_labels = 3
        self.sentiment_id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.sentiment_clf = FeedForwardMLP(
            input_dim=self.embed_dim, output_dim=self.num_sentiment_labels, hidden_dim=192
        )

    def forward_pass(self, sentences: List[str], task: str = 'classify') -> dict[str, dict]:
        """
        Compute the embeddings for input sentences and pass to prediction heads.
        :param sentences: List of strings of text input
        :param task: Str specifying which task to perform ('classify', 'sentiment', 'all')
        :return: 2 Tensors of shape (384, num_samples), one for text
          classification and one for sentiment
        """
        if task not in ('classify', 'sentiment', 'all'):
            raise ValueError(f'Task type ({task}) not supported.')

        sentence_embeddings = self.encode(sentences, convert_to_tensor=True, output_value='sentence_embedding')
        output = {}
        # Do text classification
        if task in ('classify', 'all'):
            output.update({'clf_output': self.text_clf.forward(sentence_embeddings)})

        # Do sentiment analysis
        if task in ('sentiment', 'all'):
            output.update({'sentiment_output': self.sentiment_clf.forward(sentence_embeddings)})

        return output

    def predict(self, sentences: List[str]):

        forward_outputs = self.forward_pass(sentences)

        # Class labeling

        return forward_outputs


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

    def forward(self, x: torch.Tensor): # -> dict[str, torch.Tensor]:
        # apply each layer sequentially
        x = self.hidden(x)
        x = self.gelu(x)  # Same activation as embedding layers
        logits = self.final(x)
        pred_classes = torch.argmax(logits, dim=1)
        return {'logits': logits, 'pred_classes': pred_classes}


"""
BertConfig {
  "_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
  "architectures": [
    "BertModel"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 384,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 6,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.43.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
"""
