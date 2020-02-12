Sequence Classification (or Text Classification) is the NLP task of predicting a label for a sequence of words.

A sentiment classifier is an example of a sequence classification model.

Below, we'll walk through how we can use AdaptNLP's `EasySequenceClassification` module to label unstructured text with
state-of-the-art sequence classification models.


## Getting Started with `EasySequenceClassifier`

We'll first get started by importing the `EasySequenceClassifier` class from AdaptNLP.

After that, we set some example text and instantiate the classifier.

```python
from adaptnlp import EasySequenceClassifier 

example_text = '''Novetta Solutions is the best. Albert Einstein used to be employed at Novetta Solutions. 
The Wright brothers loved to visit the JBF headquarters, and they would have a chat with Albert.'''

classifier = EasySequenceClassifier()
```

### Tagging with `tag_text(text: str, model_name_or_path: str, **kwargs)`

Now that we have the classifier instantiated, we are ready to load in a sequence classification model and tag the text
with the built-in `tag_text()` method.  

This method takes in parameters: `text` and `model_name_or_path`.
 
The method returns a list of Flair's Sentence objects.

Note: Additional keyword arguments can be passed in as parameters for Flair's token tagging `predict()` method i.e. 
`mini_batch_size`, `embedding_storage_mode`, `verbose`, etc.

```python
#Predict
sentences = classifier.tag_text(example_text, model_name_or_path="en-sentiment")
```

All of Flair's pretrained sequence classifiers are available for loading through the `model_name_or_path` parameter, 
and they can be found [here](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md).

A path to a custom trained Flair Sequence Classifier can also be passed through the `model_name_or_path` param.


```python
print("Label output:\n")
for sentence in sentences:
    print(sentence.labels)
```

Note: You can run `tag_all()` with the sequence classifier, as detailed in the Token Tagging Tutorial.