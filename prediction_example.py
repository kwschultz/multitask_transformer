import torch
import pandas as pd
from model import DualClassifier

# Use apple integrated GPU if on macbook
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == '__main__':

    test_sentences = [
        'The dog ran across the grass.',
        'Why is the sky blue?',
        'The cat jumped into the weeds.',
    ]

    model = DualClassifier()
    model.to(device)
    model.eval()

    text_classes, sentiments = model.predict(test_sentences)

    data = []

    for i in range(len(test_sentences)):
        data.append({
            'sentence': test_sentences[i],
            'text_class': text_classes[i],
            'sentiment': sentiments[i]
        })

    pd.set_option('display.max_columns', 5)
    pd.set_option('display.width', 1000)

    print("Predictions using randomized classifier weights...\n")
    print(pd.DataFrame(data))




