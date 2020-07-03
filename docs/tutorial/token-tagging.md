Token tagging (or token classification) is the NLP task of assigning a label to each individual word in the
provided text.

Examples of token tagging models are Named Entity Recognition(NER) and Parts of Speech(POS) models.  With these models,
we can generate tagged entities or parts of speech from unstructured text like "Persons" and "Nouns"

Below, we'll walk through how we can use AdaptNLP's `EasytokenTagger` module to label unstructured text with
state-of-the-art token tagging models.


## Getting Started with `EasyTokenTagger`

We'll first get started by importing the `EasyTokenTagger` module from AdaptNLP.

After that, we set some example text and instantiate the tagger.

```python
from adaptnlp import EasyTokenTagger

example_text = '''Novetta Solutions is the best. Albert Einstein used to be employed at Novetta Solutions. 
The Wright brothers loved to visit the JBF headquarters, and they would have a chat with Albert.'''

tagger = EasyTokenTagger()
```

### Tagging with `tag_text(text: str, model_name_or_path: str, **kwargs)`

Now that we have the tagger instantiated, we are ready to load in a token tagging model and tag the text with 
the built-in `tag_text()` method.  

This method takes in parameters: `text` and `model_name_or_path`.
 
The method returns a list of Flair's Sentence objects.

Note: Additional keyword arguments can be passed in as parameters for Flair's token tagging `predict()` method i.e. 
`mini_batch_size`, `embedding_storage_mode`, `verbose`, etc.

```python
# Tag the string
sentences = tagger.tag_text(text=example_text, model_name_or_path="ner-ontonotes")
```

All of Flair's pretrained token taggers are available for loading through the `model_name_or_path` parameter, 
and they can be found [here](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md).

A path to a custom trained Flair Token Tagger can also be passed through the `model_name_or_path` param.

Flair's pretrained token taggers (taken from the link above):

##### English Models
| ID | Task | Training Dataset | Accuracy |
| -------------    | ------------- |------------- |------------- |
| 'ner' | 4-class Named Entity Recognition |  Conll-03  |  **93.03** (F1) |
| 'ner-ontonotes' | [18-class](https://spacy.io/api/annotation#named-entities) Named Entity Recognition |  Ontonotes  |  **89.06** (F1) |
| 'chunk' |  Syntactic Chunking   |  Conll-2000     |  **96.47** (F1) |
| 'pos' |  Part-of-Speech Tagging |  Ontonotes     |  **98.6** (Accuracy) |
| 'frame'  |   Semantic Frame Detection |  Propbank 3.0     |  **97.54** (F1) |

##### Faster Models for CPU use
| ID | Task | Training Dataset | Accuracy |
| -------------    | ------------- |------------- |------------- |
| 'ner-fast' | 4-class Named Entity Recognition |  Conll-03  |  **92.75** (F1) |
| 'ner-ontonotes-fast' | [18-class](https://spacy.io/api/annotation#named-entities) Named Entity Recognition |  Ontonotes  |  **89.27** (F1) |
| 'chunk-fast' |  Syntactic Chunking   |  Conll-2000     |  **96.22** (F1) |
| 'pos-fast' |  Part-of-Speech Tagging |  Ontonotes     |  **98.47** (Accuracy) |
| 'frame-fast'  |   Semantic Frame Detection | Propbank 3.0     |  **97.31** (F1) |

##### Multilingual Models
| ID | Task | Training Dataset | Accuracy |
| -------------    | ------------- |------------- |------------- |
| 'ner-multi' | 4-class Named Entity Recognition |  Conll-03 (4 languages)  |  **89.27**  (average F1) |
| 'ner-multi-fast' | 4-class Named Entity Recognition |  Conll-03 (4 languages)  |  **87.91**  (average F1) |
| 'ner-multi-fast-learn' | 4-class Named Entity Recognition |  Conll-03 (4 languages)  |  **88.18**  (average F1) |
| 'pos-multi' |  Part-of-Speech Tagging   |  Universal Dependency Treebank (12 languages)  |  **96.41** (average acc.) |
| 'pos-multi-fast' |  Part-of-Speech Tagging |  Universal Dependency Treebank (12 languages)  |  **92.88** (average acc.) |

##### German Models
| ID | Task | Training Dataset | Accuracy | Contributor |
| -------------    | ------------- |------------- |------------- |------------- |
| 'de-ner' | 4-class Named Entity Recognition |  Conll-03  |  **87.94** (F1) | |
| 'de-ner-germeval' | 4+4-class Named Entity Recognition |  Germeval  |  **84.90** (F1) | |
| 'de-pos' | Part-of-Speech Tagging |  UD German - HDT  |  **98.33** (Accuracy) | |
| 'de-pos-fine-grained' | Part-of-Speech Tagging |  German Tweets  |  **93.06** (Accuracy) | [stefan-it](https://github.com/stefan-it/flair-experiments/tree/master/pos-twitter-german) |

##### Other Languages
| ID | Task | Training Dataset | Accuracy | Contributor |
| -------------    | ------------- |------------- |------------- |------------- |
| 'fr-ner' | Named Entity Recognition |  [WikiNER (aij-wikiner-fr-wp3)](https://github.com/dice-group/FOX/tree/master/input/Wikiner)  |  **95.57** (F1) | [mhham](https://github.com/mhham) |
| 'nl-ner' | Named Entity Recognition |  [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)  |  **89.56** (F1) | [stefan-it](https://github.com/stefan-it/flair-experiments/tree/master/conll2002-ner-dutch) |
| 'da-ner' | Named Entity Recognition |  [Danish NER dataset](https://github.com/alexandrainst/danlp)  |   | [AmaliePauli](https://github.com/AmaliePauli) |
| 'da-pos' | Named Entity Recognition |  [Danish Dependency Treebank](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md)  |  | [AmaliePauli](https://github.com/AmaliePauli) |

Now that the text has been tagged, take the returned sentences and see your results:

```python
# See Results
print("List string outputs of tags:\n")
for sen in sentences:
    print(sen.to_tagged_string())
```
<details class = "summary">
<summary>Output</summary>
```
Novetta <B-ORG> Solutions <E-ORG> is the best . Albert <B-PERSON> Einstein <E-PERSON> used to be employed at Novetta <B-ORG> Solutions <E-ORG> . The Wright <S-PERSON> brothers loved to visit the JBF <S-ORG> headquarters , and they would have a chat with Albert <S-PERSON> .
```
</details>

If you want just the entities, you can run the below but you'll need to specify the `label_type` as "ner" or "pos" etc.
(more information can be found in Flair's documentation):

```python
print("List entities tagged:\n")
for sen in sentences:
    for entity in sen.get_spans(label_type="ner"):
        print(entity)
```
<details class = "summary">
<summary>Output</summary>
```python
Span [1,2]: "Novetta Solutions"     [− Labels: ORG (0.9644)]
Span [7,8]: "Albert Einstein"       [− Labels: PERSON (0.9969)]
Span [14,15]: "Novetta Solutions"   [− Labels: ORG (0.9796)]
Span [18]: "Wright"                 [− Labels: PERSON (0.9995)]
Span [24]: "JBF"                    [− Labels: ORG (0.9898)]
Span [34]: "Albert"                 [− Labels: PERSON (0.9999)]
```
</details>

Here are some additional label_typess that support some of Flair's pre-trained token taggers:

| label_types | Description |
| -------------    | ------------- |
| 'ner' | For Named Entity Recognition tagged text |
| 'pos' | For Parts of Speech tagged text |
| 'np' | For Syntactic Chunking tagged text |

NOTE: You can add your own label_typess when running the sequence classifier trainer in AdaptNLP.

### Tagging with `tag_all(text: str, model_name_or_path: str, **kwargs)`

As you tag text with multiple pretrained token tagging models, your tagger will have multiple models loaded...memory
permitting.

You can then use the built-in `tag_all()` method to tag your text with all models that are currently loaded in your
tagger.  See an example below:


```python
from adaptnlp import EasyTokenTagger

example_text = '''Novetta Solutions is the best. Albert Einstein used to be employed at Novetta Solutions. 
The Wright brothers loved to visit the JBF headquarters, and they would have a chat with Albert.'''

# Load models by tagging text
tagger = EasyTokenTagger()
tagger.tag_text(text=example_text, model_name_or_path="ner-ontonotes")
tagger.tag_text(text=example_text, model_name_or_path="pos")

# Now that the "pos" and "ner-ontonotes" models are loaded, run tag_all()
sentences = tagger.tag_all(text=example_text)
```

Now we can see below that you get a list of Flair sentences tagged with the "ner-ontonotes" AND "pos" model:

```python
print("List entities tagged:\n")
for sen in sentences:
    for entity in sen.get_spans(label_type="pos"):
        print(entity)
```
<details class = "summary">
<summary>Output </summary>
```python
Span [1]: "Novetta"       [− Labels: NNP (0.9998)]
Span [2]: "Solutions"     [− Labels: NNPS (0.8235)]
Span [3]: "is"            [− Labels: VBZ (1.0)]
Span [4]: "the"           [− Labels: DT (1.0)]
Span [5]: "best"          [− Labels: JJS (0.9996)]
Span [6]: "."             [− Labels: . (0.9995)]
Span [7]: "Albert"        [− Labels: NNP (1.0)]
Span [8]: "Einstein"      [− Labels: NNP (1.0)]
Span [9]: "used"          [− Labels: VBD (0.9981)]
Span [10]: "to"           [− Labels: TO (0.9999)]
Span [11]: "be"           [− Labels: VB (1.0)]
Span [12]: "employed"     [− Labels: VBN (0.9971)]
Span [13]: "at"           [− Labels: IN (1.0)]
Span [14]: "Novetta"      [− Labels: NNP (1.0)]
Span [15]: "Solutions"    [− Labels: NNPS (0.6877)]
Span [16]: "."            [− Labels: . (0.5807)]
Span [17]: "The"          [− Labels: DT (1.0)]
Span [18]: "Wright"       [− Labels: NNP (0.9999)]
Span [19]: "brothers"     [− Labels: NNS (1.0)]
Span [20]: "loved"        [− Labels: VBD (1.0)]
Span [21]: "to"           [− Labels: TO (0.9994)]
Span [22]: "visit"        [− Labels: VB (1.0)]
Span [23]: "the"          [− Labels: DT (1.0)]
Span [24]: "JBF"          [− Labels: NNP (1.0)]
Span [25]: "headquarters" [− Labels: NN (0.9325)]
Span [26]: ","            [− Labels: , (1.0)]
Span [27]: "and"          [− Labels: CC (1.0)]
Span [28]: "they"         [− Labels: PRP (1.0)]
Span [29]: "would"        [− Labels: MD (1.0)]
Span [30]: "have"         [− Labels: VB (1.0)]
Span [31]: "a"            [− Labels: DT (1.0)]
Span [32]: "chat"         [− Labels: NN (1.0)]
Span [33]: "with"         [− Labels: IN (1.0)]
Span [34]: "Albert"       [− Labels: NNP (1.0)]
Span [35]: "."            [− Labels: . (1.0)]
```
</details>

```python
print("List entities tagged:\n")
for sen in sentences:
    for entity in sen.get_spans(label_type="ner"):
        print(entity)

```
<details class = "summary">
<summary>Output </summary>
```python
Span [1,2]: "Novetta Solutions"   [− Labels: ORG (0.9644)]
Span [7,8]: "Albert Einstein"     [− Labels: PERSON (0.9969)]
Span [14,15]: "Novetta Solutions" [− Labels: ORG (0.9796)]
Span [18]: "Wright"               [− Labels: PERSON (0.9995)]
Span [24]: "JBF"                  [− Labels: ORG (0.9898)]
Span [34]: "Albert"               [− Labels: PERSON (0.9999)]
```
</details>


