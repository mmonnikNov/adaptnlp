# What is Sequence Classification?

Sequence Classification (or Text Classification) is the NLP task of predicting a label for a sequence of words.

For example, a string of `That movie was terrible because the acting was bad` could be tagged with a label of `negative`. A string of `That movie was great because the acting was good` could be tagged with a label of `positive`.

A model that can predict sentiment from text is called a sentiment classifier, which is an example of a sequence classification model.

##### Below, we'll walk through how we can use AdaptNLP's EasySequenceClassification module to easily do the following:
1. Load pre-trained models and tag data using mini-batched inference
2. Train and fine-tune a pre-trained model on your own dataset
3. Evaluate your model

# 1. Load pre-trained models and tag data using mini-batched inference

We'll first get started by importing the EasySequenceClassifier class from AdaptNLP and instantiating the
`EasySequenceClassifier` class object.


```python
from adaptnlp import EasySequenceClassifier
from pprint import pprint

classifier = EasySequenceClassifier()
```

You can dynamically load models as you run inference.

Let's check out the Hugging Face's model repository for some pre-trained sequence classification models that some wonderful people have uploaded. The repository can be found [here](https://huggingface.co/models?search=&sort=downloads&filter=text-classification)

Let's tag some text with a model that [NLP Town](https://www.nlp.town/) has trained called `nlptown/bert-base-multilingual-uncased-sentiment`.

This is a multi-lingual model that predicts how many stars (1-5) a text review has given a product. More information can be found via. the Transformers model card [here](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)


```python
# Inference
example_text = "This didn't work at all"

sentences = classifier.tag_text(
    text=example_text,
    model_name_or_path="nlptown/bert-base-multilingual-uncased-sentiment",
    mini_batch_size=1,
)

print("Tag Score Outputs:\n")
for sentence in sentences:
    pprint({sentence.to_original_text(): sentence.labels})
```
<details class = "summary">
<summary>Output</summary>
```python
    2020-08-31 02:21:25,011 loading file nlptown/bert-base-multilingual-uncased-sentiment


    Predicting text: 100%|██████████| 1/1 [00:00<00:00, 98.67it/s]

    Tag Score Outputs:
    
    {"This didn't work at all": [1 star (0.8421),
                                 2 stars (0.1379),
                                 3 stars (0.018),
                                 4 stars (0.0012),
                                 5 stars (0.0007)]}
```
</details>



```python
multiple_text = ["This didn't work well at all.",
                 "I really liked it.",
                 "It was really useful.",
                 "It broke after I bought it."]

sentences = classifier.tag_text(
    text=multiple_text,
    model_name_or_path="nlptown/bert-base-multilingual-uncased-sentiment",
    mini_batch_size=2
)

print("Tag Score Outputs:\n")
for sentence in sentences:
    pprint({sentence.to_original_text(): sentence.labels})
```

<details class = "summary">
<summary>Output</summary>
```python
    Predicting text: 100%|██████████| 2/2 [00:00<00:00, 131.57it/s]

    Tag Score Outputs:
    
    {"This didn't work well at all.": [1 star (0.622),
                                       2 stars (0.3356),
                                       3 stars (0.0403),
                                       4 stars (0.0016),
                                       5 stars (0.0005)]}
    {'I really liked it.': [1 star (0.0032),
                            2 stars (0.0048),
                            3 stars (0.054),
                            4 stars (0.4813),
                            5 stars (0.4567)]}
    {'It was really useful.': [1 star (0.006),
                               2 stars (0.0093),
                               3 stars (0.0701),
                               4 stars (0.4136),
                               5 stars (0.501)]}
    {'It broke after I bought it.': [1 star (0.4489),
                                     2 stars (0.3935),
                                     3 stars (0.1416),
                                     4 stars (0.0121),
                                     5 stars (0.0039)]}
```

</details>

    


!!!note
    The output is going to be a probility distribution of what the text should be tagged. If you're running this on a GPU, you can specify the `mini_batch_size` parameter to run mini-batch inference against your data for faster run time.

You can set `model_name_or_path` to any of Transformer's or Flair's pre-trained sequence classification models. Transformers models are again located [here](https://huggingface.co/models). You can also pass in the path of a custom trained Transformers model.

Let's tag some text with another model, specifically Oliver Guhr's German sentiment model called `oliverguhr/german-sentiment-bert`.


```python
# Predict
german_text = ["Das hat überhaupt nicht gut funktioniert.",
               "Ich mochte es wirklich.",
               "Es war wirklich nützlich.",
               "Es ist kaputt gegangen, nachdem ich es gekauft habe."]
sentences = classifier.tag_text(
    german_text,
    model_name_or_path="oliverguhr/german-sentiment-bert",
    mini_batch_size=1
)

print("Tag Score Outputs:\n")
for sentence in sentences:
    pprint({sentence.to_original_text(): sentence.labels})
```

<details class = "summary">
<summary>Output</summary>
```python
    2020-08-31 02:21:39,109 loading file oliverguhr/german-sentiment-bert


    Predicting text: 100%|██████████| 4/4 [00:00<00:00, 132.76it/s]

    Tag Score Outputs:
    
    {'Das hat überhaupt nicht gut funktioniert.': [positive (0.0008),
                                                   negative (0.9991),
                                                   neutral (0.0)]}
    {'Ich mochte es wirklich.': [positive (0.7023),
                                 negative (0.2029),
                                 neutral (0.0947)]}
    {'Es war wirklich nützlich.': [positive (0.9813),
                                   negative (0.0184),
                                   neutral (0.0002)]}
    {'Es ist kaputt gegangen, nachdem ich es gekauft habe.': [positive (0.0042),
                                                              negative (0.9957),
                                                              neutral (0.0001)]}
```
</details>


    


Don't forget you can still quickly run inference with the multi-lingual review sentiment model you loaded in earlier(memory permitting)! Just change the `model_name_or_path` param to the model you used before.


```python
# Predict
german_text = ["Das hat überhaupt nicht gut funktioniert.",
               "Ich mochte es wirklich.",
               "Es war wirklich nützlich.",
               "Es ist kaputt gegangen, nachdem ich es gekauft habe."]
sentences = classifier.tag_text(
    german_text,
    model_name_or_path="nlptown/bert-base-multilingual-uncased-sentiment",
    mini_batch_size=1
)

print("Tag Score Outputs:\n")
for sentence in sentences:
    pprint({sentence.to_original_text(): sentence.labels})
```

<details class = "summary">
<summary>Output</summary>
```python
    Predicting text: 100%|██████████| 4/4 [00:00<00:00, 107.33it/s]

    Tag Score Outputs:
    
    {'Das hat überhaupt nicht gut funktioniert.': [1 star (0.7224),
                                                   2 stars (0.2326),
                                                   3 stars (0.0418),
                                                   4 stars (0.0024),
                                                   5 stars (0.0008)]}
    {'Ich mochte es wirklich.': [1 star (0.0092),
                                 2 stars (0.0097),
                                 3 stars (0.0582),
                                 4 stars (0.3038),
                                 5 stars (0.6191)]}
    {'Es war wirklich nützlich.': [1 star (0.0124),
                                   2 stars (0.0158),
                                   3 stars (0.0853),
                                   4 stars (0.3754),
                                   5 stars (0.5111)]}
    {'Es ist kaputt gegangen, nachdem ich es gekauft habe.': [1 star (0.5459),
                                                              2 stars (0.3205),
                                                              3 stars (0.12),
                                                              4 stars (0.0104),
                                                              5 stars (0.0032)]}
```
</details>


    


Let's release the german sentiment model to free up some memory for our next step...training! 


```python
classifier.release_model(model_name_or_path="oliverguhr/german-sentiment-bert")
```

# 2. Train and fine-tune a pre-trained model on your own dataset

Let's imagine you have your own dataset with text/label pairs you'd like to create a sequence classification model for.

With the easy sequence classifier, you can take advantage of transfer learning by fine-tuning pre-trained models on your own custom datasets.

!!!note
    The `EasySequenceClassifier` is integrated heavily with the `nlp.Dataset` and `transformers.Trainer` class objects, so please check out the [nlp](https://huggingface.co/nlp) and [transformers](https://huggingface.co/transformers) documentation for more information.

We'll first need a "custom" dataset to start training our model. Our `EasySequenceClassifier.train()` method can run with either `nlp.Dataset` objects or CSV data file paths. Since the nlp library makes it so easy, we'll use the `nlp.load_dataset()` method to load in the IMDB Sentiment dataset. We'll show an example with a CSV later. 


```python
from nlp import load_dataset

train_dataset, eval_dataset = load_dataset('imdb', split=['train', 'test'])
# Uncomment below if you want to use less data so you don't spend an hour+ on training and evaluation
train_dataset, eval_dataset = load_dataset('imdb', split=['train[:1%]', 'test[:1%]'])

pprint(vars(train_dataset.info))
```
<details class = "summary">
<summary>Output</summary>
```python
    {'builder_name': 'imdb',
     'citation': '@InProceedings{maas-EtAl:2011:ACL-HLT2011,\n'
                 '  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  '
                 'Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, '
                 'Christopher},\n'
                 '  title     = {Learning Word Vectors for Sentiment Analysis},\n'
                 '  booktitle = {Proceedings of the 49th Annual Meeting of the '
                 'Association for Computational Linguistics: Human Language '
                 'Technologies},\n'
                 '  month     = {June},\n'
                 '  year      = {2011},\n'
                 '  address   = {Portland, Oregon, USA},\n'
                 '  publisher = {Association for Computational Linguistics},\n'
                 '  pages     = {142--150},\n'
                 '  url       = {http://www.aclweb.org/anthology/P11-1015}\n'
                 '}\n',
     'config_name': 'plain_text',
     'dataset_size': 133190346,
     'description': 'Large Movie Review Dataset.\n'
                    'This is a dataset for binary sentiment classification '
                    'containing substantially more data than previous benchmark '
                    'datasets. We provide a set of 25,000 highly polar movie '
                    'reviews for training, and 25,000 for testing. There is '
                    'additional unlabeled data for use as well.',
     'download_checksums': {'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz': {'checksum': 'c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe',
                                                                                               'num_bytes': 84125825}},
     'download_size': 84125825,
     'features': {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], names_file=None, id=None),
                  'text': Value(dtype='string', id=None)},
     'homepage': 'http://ai.stanford.edu/~amaas/data/sentiment/',
     'license': '',
     'post_processed': PostProcessedInfo(features=None, resources_checksums={'train': {}, 'test': {}, 'unsupervised': {}, 'train[:10%]': {}, 'train[:1%]': {}, 'test[:1%]': {}}),
     'post_processing_size': 0,
     'size_in_bytes': 217316171,
     'splits': {'test': SplitInfo(name='test', num_bytes=32650697, num_examples=25000, dataset_name='imdb'),
                'train': SplitInfo(name='train', num_bytes=33432835, num_examples=25000, dataset_name='imdb'),
                'unsupervised': SplitInfo(name='unsupervised', num_bytes=67106814, num_examples=50000, dataset_name='imdb')},
     'supervised_keys': None,
     'version': 1.0.0}


Let's take a brief look at what the IMDB Sentiment dataset looks like. We can see that the label column has two classes of 0 and 1. You can see the name of the classes mapped to the integers with `train_dataset.features["names"]`.
```
</details>

```python
train_dataset.set_format(type="pandas", columns=["text", "label"])
train_dataset[:]
```



<details class = "summary">
<summary>Output</summary>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bromwell High is a cartoon comedy. It ran at t...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Homelessness (or Houselessness as George Carli...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Brilliant over-acting by Lesley Ann Warren. Be...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>This is easily the most underrated film inn th...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>This is not the typical Mel Brooks film. It wa...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>245</th>
      <td>1</td>
      <td>That hilarious line is typical of what these n...</td>
    </tr>
    <tr>
      <th>246</th>
      <td>1</td>
      <td>Faith and Mortality... viewed through the lens...</td>
    </tr>
    <tr>
      <th>247</th>
      <td>1</td>
      <td>The unlikely duo of Zero Mostel and Harry Bela...</td>
    </tr>
    <tr>
      <th>248</th>
      <td>1</td>
      <td>*some spoilers*&lt;br /&gt;&lt;br /&gt;I was pleasantly su...</td>
    </tr>
    <tr>
      <th>249</th>
      <td>1</td>
      <td>... and I DO mean it. If not literally (after ...</td>
    </tr>
  </tbody>
</table>
<p>250 rows × 2 columns</p>
</div>

</details>


```python
# We just run this to reformat back to a 'python' dataset
train_dataset.set_format(columns=["text", "label"])
```

Uncomment below to see training done with CSV files. The cell below will just save the `nlp.Dataset` objects you have in `train_dataset` and `eval_dataset` as CSVs and will train the model with the CSV file paths. Ignore to just continue to training.


```python
#train_dataset.set_format(type="pandas", columns=["text", "label"])
#eval_dataset.set_format(type="pandas", columns=["text", "label"])

#train_dataset[:].to_csv("./IMDB train.csv", index=False)
#eval_dataset[:].to_csv("./IMDB eval.csv", index=False)

#train_dataset = "./IMDB train.csv"
#eval_dataset = "./IMDB eval.csv"
```

One of the first things we'll need to specify before we start training are the training arguments. Training arguments consist mainly of the hyperparameters we want to provide the model. These may include batch size, initial learning rate, number of epochs, etc.

We will be using the `transformers.TrainingArguments` data class to store our training args. These are compatible with the `transformers.Trainer` as well as AdaptNLP's train methods. For more documention on the `TrainingArguments` class, please look [here](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments). There are a lot of arguments available, but we will pass in the important args and use default values for the rest.

The training arguments below specify the output directory for you model and checkpoints.


```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./models',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./logs',
    save_steps=100
)
```

Now we can run the built-in `train()` method by passing in the training arguments. The training method will also be where you specify your data arguments which include the your train and eval datasets, the pre-trained model ID (this should have been loaded from your earlier cells, but can be loaded dynamically), text column name, label column name, and ordered label names (only required if loading in paths to CSV data file for dataset args).

Please checkout AdaptNLP's package reference for more information [here](https://novetta.github.io/adaptnlp/class-api/sequence-classifier-module.html).


```python
classifier.train(training_args=training_args,
                 train_dataset=train_dataset,
                 eval_dataset=eval_dataset,
                 model_name_or_path="nlptown/bert-base-multilingual-uncased-sentiment",
                 text_col_nm="text",
                 label_col_nm="label",
                 label_names=["positive","negative"]
                )
```

# Evaluate your model 

After training, you can evaluate the model with the eval dataset you passed in for training.


```python
classifier.evaluate(model_name_or_path="nlptown/bert-base-multilingual-uncased-sentiment")
```

Now you can see it's a little weird that we're still using the `model_name_or_path` of the pre-trained model we fine-tuned and took advantage of via. transfer learning. We can release the model we've fine-tuned, and then load it back in using the directory that we've serialized the fine-tuned model. 


```python
classifier.release_model(model_name_or_path="nlptown/bert-base-multilingual-uncased-sentiment")
```


```python
sentences = classifier.tag_text(
    multiple_text,
    model_name_or_path="./models",
    mini_batch_size=1
)

print("Tag Score Outputs:\n")
for sentence in sentences:
    pprint({sentence.to_original_text(): sentence.labels})
```

<details class = "summary">
<summary>Output</summary>
```python
    Predicting text: 100%|██████████| 4/4 [00:00<00:00, 122.16it/s]

    Tag Score Outputs:
    
    {"This didn't work well at all.": [neg (0.7344), pos (0.2656)]}
    {'I really liked it.': [neg (0.2935), pos (0.7065)]}
    {'It was really useful.': [neg (0.3237), pos (0.6763)]}
    {'It broke after I bought it.': [neg (0.6209), pos (0.3791)]}
```
</details>


    


And we're done!