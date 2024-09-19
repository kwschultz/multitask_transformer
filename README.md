# multitask_transformer
Implementing a custom Sentence Transformer neural network for multi-task learning using PyTorch and Python.

<!-- `Note:` Will implementing a customized version of the MiniLM Sentence Transformer due to its balance of performance and efficiency. -->


### Goals
Implement Encoder-only architecture for model inference, deferring training for later follow up. 

`Goal 1:` Encode input sentences into fixed length embeddings. <br>
`Goal 2:` Expand architecture to support multi-task learning. 
- `Task A:` Text Classification
- `Task B:` NER, or Sentiment Analysis etc. (TODO: Select one)

### Virtual Environment Setup
- Use Conda to create virtual environment and install requirements. 
- Use Bash to run [conda_env_setup.sh](conda_env_setup.sh)

## Discussion

### 1. Discuss any choices you had to make regarding the model architecture outside of the transformer backbone.

#### 1A: 
Example embeddings for a few sentences, and their corresponding cosine similarities:
```commandline
> python test_embeddings.py
```
...
```
Embeddings size: torch.Size([3, 384])
   cosine_sim                     sentence_1                      sentence_2
1    0.376084  The dog ran across the grass.  The cat jumped into the weeds.
0    0.069512  The dog ran across the grass.            Why is the sky blue?
2    0.021868           Why is the sky blue?  The cat jumped into the weeds.
```

#### 1B: 
I chose MiniLM as the specific SentenceTransformer (based on the BERT architecture). This model has a good tradeoff between quality contextual embeddings while being computationally efficient. If the assignment was to implement from scratch, that would have taken longer than the specified 2 hours. 



### 2. Implementing multi-task learning/prediction 


<!-- Instructions for project

Step 1: Implement a Sentence Transformer Model
● Implement a sentence transformer model using any deep learning framework of your
choice. This model should be able to encode input sentences into fixed-length
embeddings.
● Test your implementation with a few sample sentences and showcase the obtained
embeddings.
● Discuss any choices you had to make regarding the model architecture outside of the
transformer backbone
Step 2: Multi-Task Learning Expansion
Expand the sentence transformer model architecture to handle a multi-task learning setting.
● Task A: Sentence Classification
○ Implement a task-specific head for classifying sentences into predefined classes
○ Classify sentences into predefined classes (you can make these up).
● Task B: [Choose an Additional NLP Task]
○ Implement a second task-specific head for a different NLP task, such as Named
Entity Recognition (NER) or Sentiment Analysis (you can make the labels up).
● Discuss the changes made to the architecture to support multi-task learning.
Note that it’s not required to actually train the multi-task learning model or implement a training
loop. The focus is on implementing a forward pass that can accept an input sentence and output
predictions for each task that you define.

Step 3: Discussion Questions
1. Consider the scenario of training the multi-task sentence transformer that you
implemented in Task 2. Specifically, discuss how you would decide which portions of the
network to train and which parts to keep frozen.
For example,
● When would it make sense to freeze the transformer backbone and only train the
task specific layers?
● When would it make sense to freeze one head while training the other?
2. Discuss how you would decide when to implement a multi-task model like the one in this
assignment and when it would make more sense to use two completely separate models
for each task.
3. When training the multi-task model, assume that Task A has abundant data, while Task
B has limited data. Explain how you would handle this imbalance.

-->


