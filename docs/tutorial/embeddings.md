Embeddings for NLP are the vector representations of unstructured text.

Examples of applications of Word Embeddings are downstream NLP task model training and similarity search. 

Below, we'll walk through how we can use AdaptNLP's `EasyWordEmbeddings`, `EasyStackedEmbeddings`, and 
`EasyDocumentEmbeddings` classes.

## Available Language Models

Huggingface's Transformer's model key shortcut names can be found [here](https://huggingface.co/transformers/pretrained_models.html).

The key shortcut names for their public model-sharing repository are available [here](https://huggingface.co/models) as of
v2.2.2 of the Transformers library.

Below are the available transformers model architectures for use as an encoder:

| Transformer Model|
| -------------    |
| ALBERT |
| DistilBERT |
| BERT |
| CamemBERT |
| RoBERTa |
| GPT |
| GPT2 |
| XLNet |
| TransformerXL |
| XLM |
| XLMRoBERTa |

You can also use Flair's `FlairEmbeddings` who's model key shortcut names are located [here](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md)

You can also use AllenNLP's `ELMOEmbeddings` who's model key shortcut names are located [here](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/ELMO_EMBEDDINGS.md)

## Getting Started with `EasyWordEmbeddings`

With `EasyWordEmbeddings`, you can load in a language model and produce contextual embeddings with text input.

You can look at each word's embeddings which have been contextualized by their surrounding text, meaning embedding
outputs will change for the same word depending on the text as a whole.

Below is an example of producing word embeddings from OpenAI's GPT2 language model.


```python
from adaptnlp import EasyWordEmbeddings

example_text = "This is Albert.  My last name is Einstein.  I like physics and atoms."

# Instantiate embeddings tagger
embeddings = EasyWordEmbeddings()

# Get GPT2 embeddings of example text... A list of flair Sentence objects are generated
sentences = embeddings.embed_text(example_text, model_name_or_path="gpt2")
# Iterate through to access the embeddings
for token in sentences[0]:
    print(token.get_embedding())
    break
```

<details class = "summary">
<summary>Output</summary>
```python
tensor([-0.1524, -0.0703,  0.5778,  ..., -0.3797, -0.3565,  2.4139])
```
</details>

## Getting Started with `EasyStackedEmbeddings`

Stacked embeddings are a simple yet important concept pointed out by Flair that can help produce state-of-the-art
results in downstream NLP models.

It produces contextualized word embeddings like `EasyWordEmbeddings`, except the embeddings are the concatenation of
tensors from multiple language model. 

Below is an example of producing stacked word embeddings from the BERT base cased and XLNet base cased language
models.  `EasyStackedEmbeddings` take in a variable number of key shortcut names to pre-trained language models.

```python
from adaptnlp import EasyStackedEmbeddings

# Instantiate stacked embeddings tagger
embeddings = EasyStackedEmbeddings("bert-base-cased", "xlnet-base-cased")

# Run the `embed_stack` method to get the stacked embeddings outlined above
sentences = embeddings.embed_text(example_text)
# Iterate through to access the embeddings
for token in sentences[0]:
    print(token.get_embedding())
    break
```
<details class = "summary">
<summary>Output</summary>
```python
tensor([ 0.5918, -0.4142,  1.0203,  ..., -0.1045, -1.2841,  0.0192])
```
</details>

## Getting Started with `EasyDocumentEmbeddings`

Document embeddings you can load in a variable number of language models, just like in stacked embeddings, and produce
and embedding for an entire text.  Unlike `EasyWordEmbeddings` and `EasyStackedEmbeddings`, `EasyDocumentEmbeddings`
will produce one contextualized embedding for a sequence of words using the pool or RNN method provided by Flair.

If you are familiar with using Flair's RNN document embeddings, you can pass in hyperparameters through the `config`
parameter when instantiating an `EasyDocumentEmbeddings` object.

Below is an example of producing an embedding from the entire text using the BERT base cased and XLNet base 
cased language models.  We also show the embeddings you get using the pool or RNN method.


```python
from adaptnlp import EasyDocumentEmbeddings

# Instantiate document embedder with stacked embeddings
embeddings = EasyDocumentEmbeddings("bert-base-cased", "xlnet-base-cased")

# Document Pool embedding...Instead of a list of flair Sentence objects, we get one Sentence object: the document
text = embeddings.embed_pool(example_text)
#get the text/document embedding
text[0].get_embedding()
```
<details class = "summary">
<summary>Output</summary>
```python
tensor([ 0.4216,  0.0123,  0.3136,  ..., -0.0683, -0.3761, -0.0974],
       grad_fn=<CatBackward>)
```
</details>

```python
# Now again but with Document RNN embedding
text = embeddings.embed_rnn(example_text)
#get the text/document embedding
text[0].get_embedding()
```
<details class = "summary">
<summary>Output</summary>
```python
tensor([ 4.0643e-02,  4.7823e-01,  3.5992e-01, -6.5744e-01,  2.5690e-01,
        -2.2250e-02,  6.6651e-01, -1.4607e-01,  4.8427e-01,  6.3852e-01,
         9.8436e-02, -1.4234e-01, -6.1204e-01,  4.4708e-01,  2.4172e-01,
         2.4852e-01, -1.5021e-01,  5.1846e-01, -1.2435e-01,  1.1078e-01,
         3.6920e-01,  2.3225e-01, -2.2924e-01, -4.9226e-02,  4.7070e-01,
        -1.3099e-01,  7.9573e-01,  2.7918e-01, -6.8034e-01, -5.7282e-01,
         2.8865e-01, -5.9626e-01,  5.1510e-01,  2.0294e-01,  3.4929e-01,
        -5.5842e-02, -4.6091e-01, -3.9273e-01, -4.6477e-01,  7.3891e-02,
         3.1949e-01, -3.3215e-01,  1.3878e-01,  2.8379e-01, -4.9557e-02,
        -4.5319e-01,  1.1646e-02, -6.0409e-02, -5.8763e-01,  8.0155e-01,
        -2.2879e-02,  2.3967e-01,  6.0385e-01, -4.1895e-01, -1.6761e-01,
         6.4883e-01,  6.1100e-01, -7.7293e-01,  1.7982e-01,  8.7999e-02,
         4.7579e-01, -2.4647e-01,  2.9902e-01, -4.4531e-01,  3.4841e-01,
        -7.9070e-01,  5.7861e-02, -1.3308e-01, -1.0392e-01,  4.7919e-01,
        -6.1978e-01, -1.7192e-01, -4.7946e-01,  4.5381e-02, -3.7442e-02,
        -6.8591e-01,  3.5243e-01, -1.9135e-01,  3.6689e-01, -2.1427e-01,
        -1.3946e-01,  2.9380e-01,  2.4939e-01,  6.5739e-02,  4.7131e-01,
        -8.2398e-01,  6.1843e-02, -5.4207e-01, -4.3683e-01, -1.5192e-01,
        -1.5242e-02, -7.6256e-01, -4.8683e-01,  1.7045e-01,  1.0848e-01,
         3.5006e-01, -2.8152e-01, -7.3525e-02,  1.7871e-01,  4.3365e-01,
         2.8071e-01, -1.7845e-01, -4.7001e-01,  3.0485e-01, -3.1472e-01,
         7.8487e-01, -8.2343e-01,  2.5580e-01,  6.2897e-02, -5.3286e-01,
         1.0242e-01, -8.8470e-02, -9.5680e-02,  8.5138e-01, -3.2669e-03,
        -1.9355e-01, -1.0739e-01,  1.0788e-01,  4.6164e-01, -6.7108e-02,
         2.0659e-01,  6.9547e-01,  5.2934e-01, -4.9506e-01, -5.4222e-01,
         4.2463e-01,  3.7806e-01, -3.4682e-01,  2.4633e-02,  4.5355e-01,
        -1.9444e-01,  7.5605e-01, -1.6126e-01,  7.1877e-01, -1.9557e-01,
        -2.2612e-01, -5.0139e-02, -1.3550e-01, -1.1433e-01, -8.1717e-01,
        -1.9096e-01,  6.8815e-02,  6.9301e-02, -2.7783e-01, -5.6060e-02,
         2.3175e-01, -4.5415e-01, -8.8416e-02,  5.2196e-02,  3.6615e-01,
        -2.9025e-01,  1.3258e-01, -4.9883e-01,  2.2678e-01, -4.2092e-01,
        -7.2251e-01, -4.0375e-01, -1.5807e-02,  4.6092e-01,  2.9596e-01,
         3.6077e-01,  1.9079e-01, -2.5271e-01, -2.7760e-02,  3.6855e-01,
         3.8165e-01,  6.0619e-03, -7.6378e-01, -3.7182e-01, -4.4542e-02,
        -2.0117e-01, -1.1995e-01,  2.3850e-01, -4.1636e-01, -4.8439e-01,
        -2.0748e-02,  5.4735e-01, -7.2940e-01, -4.1707e-01,  5.9896e-01,
         1.7213e-01, -1.3483e-01, -3.8994e-01,  3.7115e-01, -2.4966e-01,
        -2.7104e-01,  1.3207e-01,  7.1423e-02,  2.1035e-01,  4.5386e-01,
        -3.4646e-01,  1.3394e-01, -3.7041e-01, -4.2550e-01,  2.6191e-02,
         6.6384e-01,  1.2815e-01,  2.6748e-02,  5.0338e-01,  4.1966e-02,
         7.0873e-02,  4.2947e-01, -1.2464e-01, -2.1960e-02, -1.8431e-01,
         7.5072e-01, -3.0089e-02,  3.0614e-01,  2.7832e-01, -6.7883e-01,
        -4.5706e-01, -1.6099e-01,  5.7140e-01,  5.3964e-01,  1.6853e-01,
        -1.2111e-01, -7.3538e-01,  1.0851e-01, -1.8549e-01, -2.6486e-01,
        -1.3871e-02,  2.8989e-01,  8.7540e-02, -4.0214e-01,  1.9980e-02,
        -7.1209e-02,  4.6514e-01,  2.3598e-01,  6.6215e-01, -5.3153e-01,
        -4.3674e-02,  9.7224e-02, -1.9030e-01, -7.5050e-01,  4.6526e-01,
         4.3002e-01, -5.4262e-01, -3.7726e-01,  2.8196e-01, -1.6574e-01,
        -7.0038e-02, -1.9054e-01,  2.9857e-01, -1.0482e-01, -1.1758e-01,
         1.2275e-02,  1.6027e-01, -3.4117e-02, -2.9249e-01,  2.8828e-01,
        -1.1687e-01, -5.2637e-02,  5.3424e-01,  2.1326e-01, -1.1130e-02,
         1.1047e-01,  4.6660e-01, -6.8302e-02, -6.2710e-01, -5.3588e-01,
         5.6987e-01,  1.0222e-01,  2.4219e-02, -2.5624e-01,  8.0474e-02,
         4.1616e-01, -6.5643e-01, -6.0552e-01, -3.6263e-01, -1.0691e-01,
        -2.3464e-01, -3.3408e-01,  1.3120e-01,  9.2258e-02, -1.2690e-01,
        -3.9567e-01,  2.8039e-01,  5.4222e-02,  1.7499e-01, -8.6867e-01,
        -3.6676e-01,  3.8382e-02, -6.8972e-01, -3.3034e-01,  2.9412e-01,
         4.1795e-01, -3.7838e-01, -1.9996e-01, -7.1303e-02, -2.2892e-01,
        -1.6649e-01,  7.2984e-02, -2.6782e-01,  8.0018e-01,  4.2457e-01,
        -6.1137e-02,  1.3479e-01,  6.0753e-01, -4.9129e-01,  5.4194e-01,
         5.5168e-01,  4.2584e-01,  7.7317e-01,  3.9137e-01,  2.5688e-02,
         4.7761e-01,  1.1689e-02, -3.0685e-02, -3.9294e-01,  2.5061e-01,
        -3.3469e-01, -1.2963e-01, -6.4249e-01,  4.8470e-01,  1.0723e-01,
         4.1352e-02, -3.2012e-01, -3.1172e-02, -8.9335e-01,  1.5820e-01,
         2.4536e-01, -3.7762e-02, -3.0008e-01, -1.4802e-01,  3.1138e-01,
        -4.0120e-01,  5.2545e-01, -2.9134e-01,  1.2147e-01, -7.2720e-01,
         7.3572e-01, -5.5248e-01,  2.3775e-02, -8.5957e-02, -7.8563e-02,
        -2.3558e-01,  2.9765e-01,  2.3477e-01,  4.7899e-01, -1.3915e-01,
         2.8876e-03,  6.6184e-02, -1.9383e-01,  5.8549e-01,  2.0872e-01,
         3.7530e-01,  3.8544e-01,  1.5910e-01, -6.1431e-01,  2.3468e-01,
         4.7251e-01,  3.3820e-01,  7.2941e-01,  2.4980e-02, -5.1878e-01,
         3.8167e-01, -5.2926e-01, -8.3092e-02,  4.0554e-01, -3.2830e-01,
        -7.9680e-01,  5.0002e-01,  2.5843e-01, -1.7918e-01,  9.9184e-02,
         2.4426e-02,  1.4300e-01, -2.1614e-01, -1.9736e-01, -1.6995e-01,
        -3.8350e-01, -5.9254e-01, -7.3108e-01, -9.7870e-02,  6.9274e-01,
         2.8090e-01,  7.2932e-02, -1.8584e-01, -2.0107e-02, -3.9466e-01,
        -1.3001e-01, -4.5177e-01, -4.0892e-03,  6.3328e-01, -8.3370e-02,
         3.9102e-01,  2.0180e-01,  2.4513e-01,  2.1440e-01,  3.8041e-01,
        -3.4213e-01,  8.5629e-02, -1.9770e-01, -5.4095e-02, -5.3485e-01,
         7.2403e-02, -3.0714e-01, -2.7611e-01, -3.0493e-01, -3.8583e-01,
        -1.3889e-01, -2.8492e-01,  2.4335e-01,  3.2545e-03,  4.0507e-01,
        -2.8886e-01,  1.5966e-01,  3.5735e-01,  4.9109e-01, -1.1930e-01,
        -6.4261e-02,  3.0875e-02, -2.1206e-01,  3.6731e-01,  3.0674e-01,
        -5.4629e-01, -1.9124e-02, -3.6374e-01, -2.1023e-01,  1.7612e-01,
         6.9023e-01, -1.0726e-01, -3.1508e-01, -5.5917e-02,  4.9525e-01,
        -2.5035e-01, -4.3870e-01, -2.1269e-01,  3.6930e-01, -3.5634e-02,
        -8.2272e-01,  3.5745e-01, -2.9108e-01,  1.8137e-01,  3.2459e-01,
         6.1389e-01,  2.0270e-01,  2.9765e-01,  2.9563e-01,  3.0103e-01,
        -5.6877e-01, -2.2441e-01,  2.3133e-01, -4.2049e-01,  2.7534e-01,
        -2.6664e-01,  1.0737e-01, -3.7153e-01, -5.1736e-01,  2.5754e-01,
         3.4389e-01, -2.1162e-01,  3.9876e-01,  3.0114e-01,  3.2266e-01,
        -3.0570e-01,  1.0993e-01,  3.4368e-01,  5.7563e-01, -1.7115e-01,
         3.3226e-01, -4.0898e-01,  1.5295e-01, -6.6033e-01, -3.5574e-01,
        -4.9282e-02, -2.7427e-01,  4.8897e-01,  5.2119e-01, -2.0027e-01,
         5.6864e-01,  2.7602e-01,  2.2527e-02, -1.9639e-01, -3.1784e-01,
        -2.2034e-01, -1.6692e-01, -1.4974e-01, -1.6638e-01, -9.1813e-02,
         7.2773e-01,  1.6606e-01, -1.0737e-01,  5.5271e-01,  3.9674e-01,
         3.2050e-01, -1.2518e-01,  2.5195e-01,  3.1479e-01, -4.6130e-01,
         3.1082e-01, -3.2721e-01,  4.8215e-01,  5.8966e-01, -2.4745e-01,
         3.3863e-01,  4.0711e-01, -1.2112e-01, -7.8878e-02,  3.4396e-01,
         4.3243e-01,  1.7047e-01,  7.0417e-01,  1.5878e-01,  2.3164e-01,
         1.3155e-04,  7.7327e-01,  4.2866e-01, -4.5849e-01,  2.1252e-01,
         1.0223e-01, -4.8963e-01], grad_fn=<CatBackward>)
```
</details>