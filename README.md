# multitask_transformer
Implementing a custom Sentence Transformer neural network for multi-task learning using PyTorch and Python.

### Goals
Implement Encoder-only architecture for embedding sentences, then two independent classifier heads on top of the embeddings for multi-task predictions.  

`Goal 1:` Encode input sentences into fixed length embeddings. <br>
`Goal 2:` Expand architecture to support multi-task learning. 
- `Task A:` Text Classification
  - 6 Classes: 'sports', 'health', 'tech', 'finance', 'education', 'other'
- `Task B:` Sentiment Analysis 
  - 3 Classes: 'negative', 'neutral', 'positive'

### Virtual Environment Setup
- Use Conda to create virtual environment and install requirements. 
- Use Bash to run [conda_env_setup.sh](conda_env_setup.sh)

### Architecture 
- SentenceTransformer ('all-MiniLM-L6-v2') backbone to compute contextualized sentence embeddings. It has a good balance of performance and efficiency.
- Added two instances of a simple multi-layer perceptron for each classification head
- Each classifier MLP designed for simplicity, while being able to model complex relationships between embeddings and classes. 
  - fully connected layer from embeddings into smaller hidden representation (half the embedding size 384 -> 192)
  - GELU activation function
    - Chose nonlinear GELU function to match what was used in MiniLM backbone
  - Final fully connected layer to the classes (192 -> 6) or (192 -> 3)

### Embeddings Example
Embedding a few sentences, and their corresponding cosine similarities (relevance):
```commandline
> python embedding_example.py
```
...
```
Embeddings size: torch.Size([3, 384])
   cosine_sim                     sentence_1                      sentence_2
1    0.376084  The dog ran across the grass.  The cat jumped into the weeds.
0    0.069512  The dog ran across the grass.            Why is the sky blue?
2    0.021868           Why is the sky blue?  The cat jumped into the weeds.
```

### Multi-task Predictions 

Example text classification and sentiment predictions for a few sentences. <br>
`Note:` With randomized model weights on init, each run produces different predictions.
```commandline
> python prediction_example.py
```

```
Multi-task predictions using randomized classifier weights...

                         sentence   text_class    sentiment
0   The dog ran across the grass.         tech     positive
1            Why is the sky blue?      finance      neutral
2  The cat jumped into the weeds.       health     negative
```

<!--
### Multi-Task Model Training Considerations 

#### To begin training the multi-task sentence transformer
1. If applying to same/similar domain as the pretraining corpus, freeze the weights of the embedding layers to preserve pretraining generality (e.g. MiniLM layers)
2. If modeling a different or specialized domain, fine-tuning some of the embedding layers in addition to the classification layers can work well.  
2. To train for classification tasks, update the weights for only the classifier layers
   1. If the dataset contains all labels for each example (e.g. text class and sentiment), then training similar tasks jointly makes sense. Include both in the same loss function for regularization.
   2. If the task training is done with separate datasets or tasks are dissimilar, then train independently.
      1. E.g. For training on sentiment dataset, freeze the named entity recognition layers

#### Implementation of multi-task model or independent models
1. Prefer a multi-task single model when
   2. Tasks input data distribution is similar, and the tasks need similar context or features 
   3. Performance constraints favor speed, a multi-task model only computes the initial representation once. 
   4. Depending on the model deployment specifics, it may be less efficient to pass the input data to another model and get the response back.
2. Prefer individual task models when
   1. The tasks need to be performed at different times/cadence.
   2. The tasks are not similar or the input data for each task is different.
   3. When jointly training a multi-task model, you may observe that the gradients for each task's loss function oppose each other, leading to slower model convergence and lower performance.
-->


#### When the amount of data for each task differs greatly

Assume task B has limited data but is still similar to task A, which has ample data. 
- This scenario can be addressed with sequential task-specific fine tuning. 
    - Initially fine tuning the model for task A, then freezing those weights and fine tuning for task B.
- It is possible to explore text data augmentation (synthetic data generation) to supplement the training data for task B.
