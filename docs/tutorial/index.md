This tutorial section goes over the NLP capabilities available through AdaptNLP and how to use them.

You should ideally follow the tutorials along with the provided notebooks in the `tutorials` directory at the top
level of the AdaptNLP library.

You could also run the code snippets in these tutorials straight through the python interpreter as well.

## Install and Setup

You will first need to install AdaptNLP with Python 3.6+ using the following command:

```
pip install adaptnlp
```

AdaptNLP is largely built on top of Flair, Transformers, and PyTorch, and dependencies will be handled on install.

AdaptNLP can be used with or without GPUs.  AdaptNLP will automatically make use of GPU VRAM in environment with
CUAD-compatible NVIDIA GPUs and NVIDIA drivers installed.  GPU-less environments will run AdaptNLP modules fine as well.


## Overview of NLP Capabilities

An overview of some of the AdaptNLP capabilities via. code snippets.

A good way to make sure AdaptNLP is installed correctly is by running each of these code snippets.

Note this may take a while if models have not already been downloaded.

##### Named Entity Recognition with `EasyTokenTagger`

```python
from adaptnlp import EasyTokenTagger

## Example Text
example_text = "Novetta's headquarters is located in Mclean, Virginia."

## Load the token tagger module and tag text with the NER model 
tagger = EasyTokenTagger()
sentences = tagger.tag_text(text=example_text, model_name_or_path="ner")

## Output tagged token span results in Flair's Sentence object model
for sentence in sentences:
    for entity in sentence.get_spans("ner"):
        print(entity)

```
<details class = "summary">
 <summary>Output</summary>
```python
Span [1]: "Novetta"   [− Labels: ORG (0.9925)]
Span [7]: "Mclean"    [− Labels: LOC (0.9993)]
Span [9]: "Virginia"  [− Labels: LOC (1.0)]
```
</details>

##### English Sentiment Classifier `EasySequenceClassifier`

```python
from adaptnlp import EasySequenceClassifier 

## Example Text
example_text = "Novetta is a great company that was chosen as one of top 50 great places to work!"

## Load the sequence classifier module and classify sequence of text with the english sentiment model 
classifier = EasySequenceClassifier()
sentences = classifier.tag_text(text=example_text, model_name_or_path="en-sentiment")

## Output labeled text results in Flair's Sentence object model
for sentence in sentences:
    print(sentence.labels)

```
<details class = "summary">
<summary>Output</summary>
```python
[POSITIVE (0.9977)]
```
</details>


##### Language Model Embeddings `EasyWordEmbeddings`
```python
from adaptnlp import EasyWordEmbeddings

## Example Text
example_text = "Albert Einstein used to work at Novetta."

## Load the embeddings module and embed the tokens within the text
embeddings = EasyWordEmbeddings()
sentences = embeddings.embed_text(example_text, model_name_or_path="gpt2")

# Iterate through tokens in the sentence to access embeddings
for sentence in sentences:
    for token in sentence:
        print(token.get_embedding())
```
<details class = "summary">
<summary>Output</summary>
```python
tensor([-1.8757,  0.6195, -1.3108,  ..., -1.3787, -0.6885,  1.6934])
tensor([-0.0617, -2.3885,  2.2028,  ...,  0.2774,  0.8424, -1.5328])
tensor([-0.0480, -0.7461, -0.5282,  ...,  0.1554,  0.2542,  0.8199])
tensor([ 1.0621, -0.3834,  1.5259,  ..., -0.0937, -0.0337,  1.0316])
tensor([-0.0027, -1.6549, -1.6274,  ...,  0.3001,  0.0146, -0.1931])
tensor([ 0.6624, -0.9889,  0.6716,  ..., -0.4907,  0.5692,  0.9456])
tensor([ 0.5633,  0.4789, -0.2232,  ..., -0.1454,  0.2486,  0.5163])
```
</details>



##### Span-based Question Answering `EasyQuestionAnswering`

```python
from adaptnlp import EasyQuestionAnswering 

## Example Query and Context 
query = "What is the meaning of life?"
context = "Machine Learning is the meaning of life."
top_n = 5

## Load the QA module and run inference on results 
qa = EasyQuestionAnswering()
best_answer, best_n_answers = qa.predict_qa(query=query, context=context, n_best_size=top_n)

## Output top answer as well as top 5 answers
print(best_answer)
print(best_n_answers)
```
<details class = "summary">
<summary>Output</summary>
```python
[OrderedDict([('text', 'Machine Learning'), ('probability', 0.9924118248851219), ('start_logit', 8.646799087524414), ('end_logit', 8.419432640075684), ('start_index', 0), ('end_index', 1)]), OrderedDict([('text', 'Learning'), ('probability', 0.004796293656050888), ('start_logit', 3.314504384994507), ('end_logit', 8.419432640075684), ('start_index', 1), ('end_index', 1)]), OrderedDict([('text', 'Machine Learning is the meaning of life.'), ('probability', 0.0018383556202966893), ('start_logit', 8.646799087524414), ('end_logit', 2.1281659603118896), ('start_index', 0), ('end_index', 6)]), OrderedDict([('text', 'Machine'), ('probability', 0.0009446411263795704), ('start_logit', 8.646799087524414), ('end_logit', 1.4623442888259888), ('start_index', 0), ('end_index', 0)]), OrderedDict([('text', 'Learning is the meaning of life.'), ('probability', 8.884712150840367e-06), ('start_logit', 3.314504384994507), ('end_logit', 2.1281659603118896), ('start_index', 1), ('end_index', 6)])]
```
</detais>

<!-- 
##### Sequence Classification Training `SequenceClassifier`
```python
from adaptnlp import EasyDocumentEmbeddings, SequenceClassifierTrainer 

# Specify corpus data directory and model output directory
corpus = "Path/to/data/directory" 
OUTPUT_DIR = "Path/to/output/directory" 

# Instantiate AdaptNLP easy document embeddings module, which can take in a variable number of embeddings to make `Stacked Embeddings`.  
# You may also use custom Transformers LM models by specifying the path the the language model
doc_embeddings = EasyDocumentEmbeddings(model_name_or_path="bert-base-cased", methods = ["rnn"])

# Instantiate Sequence Classifier Trainer by loading in the data, data column map, and embeddings as an encoder
sc_trainer = SequenceClassifierTrainer(corpus=corpus, encoder=doc_embeddings, column_name_map={0: "text", 1:"label"})

# Find Learning Rate
learning_rate = sc_trainer.find_learning_rate(output_dir-OUTPUT_DIR)

# Train Using Flair's Sequence Classification Head
sc_trainer.train(output_dir=OUTPUT_DIR, learning_rate=learning_rate, max_epochs=150)


# Predict text labels with the trained model using `EasySequenceClassifier`
from adaptnlp import EasySequenceClassifier
example_text = '''Where was the Queen's wedding held? '''
classifier = EasySequenceClassifier()
sentences = classifier.tag_text(example_text, model_name_or_path=OUTPUT_DIR / "final-model.pt")
print("Label output:\n")
for sentence in sentences:
    print(sentence.labels)
```

##### Transformers Language Model Fine Tuning `LMFineTuner`

```python
from adaptnlp import LMFineTuner

# Specify Text Data File Paths
train_data_file = "Path/to/train.csv"
eval_data_file = "Path/to/test.csv"

# Instantiate Finetuner with Desired Language Model
finetuner = LMFineTuner(train_data_file=train_data_file, eval_data_file=eval_data_file, model_type="bert", model_name_or_path="bert-base-cased")
finetuner.freeze()

# Find Optimal Learning Rate
learning_rate = finetuner.find_learning_rate(base_path="Path/to/base/directory")
finetuner.freeze()

# Train and Save Fine Tuned Language Models
finetuner.train_one_cycle(output_dir="Path/to/output/directory", learning_rate=learning_rate) -->