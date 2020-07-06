A sequence classifier predicts a categorical label from the unstructured sequence of text that is provided as input.

AdaptNLP's `SequenceClassifierTrainer` uses Flair's sequence classification prediction head with Transformer and/or
Flair's contextualized embeddings.

You can specify the encoder you want to use from any of the following pretrained transformer language models provided
by Huggingface's Transformers library.  The model key shortcut names are located [here](https://huggingface.co/transformers/pretrained_models.html).

The key shortcut names of their public model-sharing repository are available [here](https://huggingface.co/models) as of
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

## Getting Started with `SequenceClassifierTrainer`

We want to start by specifying three things:
  1. `corpus`: A Flair `Corpus` data model object that contains train, test, and dev datasets.
    This can also be a path to a directory that contains `train.csv`, `test.csv`, and `dev.csv` files.
    If a path to the files is provided, you will require a `column_name_map` parameter that maps the indices of
    the `text` and `label` column headers i.e. {0: "text", 1: "label"} the colummn with text being at index 0 of the csv
  2. `output_dir`: A path to a directory to store trainer and model files
  3. `doc_embeddings`: The `EasyDocumentEmbeddings` object that has the specified key shortcut names to pretrained
    language models that the trainer will use as its encoder.

```python
from adaptnlp import EasyDocumentEmbeddings, SequenceClassifierTrainer
from flair.datasets import TREC_6

corpus = TREC_6() # Or path to directory of train.csv, test.csv, dev.csv files at "Path/to/data/directory" 

OUTPUT_DIR = "Path/to/model/output/directory" 

doc_embeddings = EasyDocumentEmbeddings("bert-base-cased", methods = ["rnn"]) # We can specify to load the pool or rnn
                                                                              # methods to avoid loading both.
```

Then we want to instantiate the trainer with the following parameters

```python
sc_configs = {
              "corpus": corpus,
              "encoder": doc_embeddings,
              "column_name_map": {0:"text",1:"label"},
              "corpus_in_memory": True,
              "predictive_head": "flair",
             }
sc_trainer = SequenceClassifierTrainer(**sc_configs)

```

<details class = "summary">
<summary>Output</summary>
```python
2020-07-02 14:25:28,470 [b'HUM', b'DESC', b'ENTY', b'NUM', b'LOC', b'ABBR']
```
</details>

We can then find the optimal learning rate with the help of the [cyclical learning rates method](https://arxiv.org/abs/1506.01186)
by Leslie Smith.

Using this along with our novel approach in [automatically extracting](https://forums.fast.ai/t/automated-learning-rate-suggester/44199?u=aychang)
an optimal learning rate, we can streamline training without pausing to manually extract the optimal learning rate.

The built-in `find_learning_rate()` will automatically reinitialize the parameteres and optimizer after running the
cyclical learning rates method.

```python
sc_lr_configs = {
        "output_dir": OUTPUT_DIR,
        "start_learning_rate": 1e-8,
        "end_learning_rate": 10,
        "iterations": 100,
        "mini_batch_size": 32,
        "stop_early": True,
        "smoothing_factor": 0.8,
        "plot_learning_rate": True,
}
learning_rate = sc_trainer.find_learning_rate(**sc_lr_configs)
```
<details class = "summary">
<summary>Output</summary>
```python
[1.5135612484362082e-08]
[1.8620871366628676e-08]
[2.2908676527677733e-08]
[2.818382931264454e-08]
[3.4673685045253164e-08]
[4.265795188015927e-08]
[5.2480746024977265e-08]
[6.456542290346554e-08]
[7.943282347242815e-08]
[9.772372209558107e-08]
[1.2022644346174127e-07]
[1.4791083881682077e-07]
[1.819700858609984e-07]
[2.2387211385683393e-07]
[2.754228703338167e-07]
[3.3884415613920264e-07]
[4.168693834703353e-07]
[5.128613839913649e-07]
[6.309573444801935e-07]
[7.762471166286916e-07]
[9.54992586021436e-07]
[1.1748975549395298e-06]
[1.4454397707459273e-06]
[1.778279410038923e-06]
[2.187761623949553e-06]
[2.6915348039269168e-06]
[3.311311214825913e-06]
[4.0738027780411255e-06]
[5.011872336272722e-06]
[6.165950018614822e-06]
[7.585775750291839e-06]
[9.332543007969913e-06]
[1.1481536214968832e-05]
[1.4125375446227536e-05]
[1.737800828749375e-05]
[2.1379620895022316e-05]
[2.6302679918953824e-05]
[3.2359365692962836e-05]
[3.981071705534974e-05]
[4.8977881936844595e-05]
[6.025595860743576e-05]
[7.413102413009174e-05]
[9.120108393559098e-05]
[0.00011220184543019637]
[0.00013803842646028855]
[0.00016982436524617435]
[0.00020892961308540387]
[0.00025703957827688637]
[0.00031622776601683794]
[0.0003890451449942807]
[0.00047863009232263854]
[0.0005888436553555893]
[0.0007244359600749906]
[0.0008912509381337464]
[0.0010964781961431862]
[0.001348962882591652]
[0.0016595869074375593]
[0.002041737944669528]
[0.002511886431509579]
[0.00309029543251359]
[0.003801893963205612]
[0.004677351412871982]
[0.005754399373371571]
[0.007079457843841382]
[0.008709635899560813]
[0.010715193052376074]
[0.013182567385564083]
[0.016218100973589285]
[0.019952623149688778]
[0.024547089156850287]
[0.030199517204020147]
[0.03715352290971724]
[0.0457088189614875]
[0.05623413251903491]
[0.06918309709189367]
[0.08511380382023769]
[0.10471285480509002]
[0.1288249551693135]
[0.1584893192461115]
[0.19498445997580477]
[0.2398832919019488]
[0.29512092266663836]
[0.3630780547701011]
[0.446683592150963]
[0.5495408738576244]
[0.6760829753919818]
[0.8317637711026711]
[1.0232929922807545]
2020-07-02 14:31:22,204 ----------------------------------------------------------------------------------------------------
2020-07-02 14:31:22,205 loss diverged - stopping early!
2020-07-02 14:31:22,364 ----------------------------------------------------------------------------------------------------
2020-07-02 14:31:22,365 learning rate finder finished - plot Path/to/model/output/directory/learning_rate.tsv
2020-07-02 14:31:22,366 ----------------------------------------------------------------------------------------------------
Learning_rate plots are saved in Path/to/model/output/directory/learning_rate.png

Recommended Learning Rate 0.016218100973589285
```
</details>

We can then kick off training below.

```python

sc_train_configs = {
        "output_dir": OUTPUT_DIR,
        "learning_rate": learning_rate,
        "mini_batch_size": 32,
        "anneal_factor": 0.5,
        "patience": 5,
        "max_epochs": 150,
        "plot_weights": False,
        "batch_growth_annealing": False,
}
sc_trainer.train(**sc_train_configs)

```
<details class = "summary">
<summary>Output</summary>
```python
2020-07-02 14:39:15,191 ----------------------------------------------------------------------------------------------------
2020-07-02 14:39:15,194 Model: "TextClassifier(
  (document_embeddings): DocumentRNNEmbeddings(
    (embeddings): StackedEmbeddings(
      (list_embedding_0): BertEmbeddings(
        (model): BertModel(
          (embeddings): BertEmbeddings(
            (word_embeddings): Embedding(28996, 768, padding_idx=0)
            (position_embeddings): Embedding(512, 768)
            (token_type_embeddings): Embedding(2, 768)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (encoder): BertEncoder(
            (layer): ModuleList(
              (0): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (1): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (2): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (3): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (4): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (5): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (6): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (7): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (8): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (9): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (10): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (11): BertLayer(
                (attention): BertAttention(
                  (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): BertIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
          )
          (pooler): BertPooler(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (activation): Tanh()
          )
        )
      )
    )
    (word_reprojection_map): Linear(in_features=3072, out_features=256, bias=True)
    (rnn): GRU(256, 512, batch_first=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (decoder): Linear(in_features=512, out_features=6, bias=True)
  (loss_function): CrossEntropyLoss()
  (beta): 1.0
  (weights): None
  (weight_tensor) None
)"
2020-07-02 14:39:15,196 ----------------------------------------------------------------------------------------------------
2020-07-02 14:39:15,197 Corpus: "Corpus: 4907 train + 545 dev + 500 test sentences"
2020-07-02 14:39:15,198 ----------------------------------------------------------------------------------------------------
2020-07-02 14:39:15,199 Parameters:
2020-07-02 14:39:15,199  - learning_rate: "0.016218100973589285"
2020-07-02 14:39:15,200  - mini_batch_size: "32"
2020-07-02 14:39:15,201  - patience: "5"
2020-07-02 14:39:15,201  - anneal_factor: "0.5"
2020-07-02 14:39:15,202  - max_epochs: "150"
2020-07-02 14:39:15,203  - shuffle: "True"
2020-07-02 14:39:15,203  - train_with_dev: "False"
2020-07-02 14:39:15,204  - batch_growth_annealing: "False"
2020-07-02 14:39:15,205 ----------------------------------------------------------------------------------------------------
2020-07-02 14:39:15,206 Model training base path: "Path/to/model/output/directory"
2020-07-02 14:39:15,206 ----------------------------------------------------------------------------------------------------
2020-07-02 14:39:15,207 Device: cpu
2020-07-02 14:39:15,208 ----------------------------------------------------------------------------------------------------
2020-07-02 14:39:15,209 Embeddings storage mode: cpu
2020-07-02 14:39:15,214 ----------------------------------------------------------------------------------------------------
2020-07-02 14:39:36,964 epoch 1 - iter 15/154 - loss 2.12653748 - samples/sec: 22.31
2020-07-02 14:39:55,974 epoch 1 - iter 30/154 - loss 2.03583712 - samples/sec: 25.44
2020-07-02 14:40:14,865 epoch 1 - iter 45/154 - loss 2.02372188 - samples/sec: 25.81
2020-07-02 14:40:35,062 epoch 1 - iter 60/154 - loss 2.03083786 - samples/sec: 23.91
2020-07-02 14:40:56,537 epoch 1 - iter 75/154 - loss 2.00187496 - samples/sec: 22.48
2020-07-02 14:41:15,410 epoch 1 - iter 90/154 - loss 1.98854279 - samples/sec: 25.77
2020-07-02 14:41:34,838 epoch 1 - iter 105/154 - loss 1.97349383 - samples/sec: 24.85
2020-07-02 14:41:55,114 epoch 1 - iter 120/154 - loss 1.96310420 - samples/sec: 23.81
2020-07-02 14:42:15,951 epoch 1 - iter 135/154 - loss 1.94268769 - samples/sec: 23.17
2020-07-02 14:42:35,886 epoch 1 - iter 150/154 - loss 1.92316744 - samples/sec: 24.23
2020-07-02 14:42:39,888 ----------------------------------------------------------------------------------------------------
2020-07-02 14:42:39,890 EPOCH 1 done: loss 1.9191 - lr 0.0162181
2020-07-02 14:43:02,558 DEV : loss 1.538955569267273 - score 0.7994
2020-07-02 14:43:02,626 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 14:43:03,149 ----------------------------------------------------------------------------------------------------
2020-07-02 14:43:26,692 epoch 2 - iter 15/154 - loss 1.69178906 - samples/sec: 20.60
2020-07-02 14:43:46,382 epoch 2 - iter 30/154 - loss 1.70241214 - samples/sec: 24.53
2020-07-02 14:44:06,135 epoch 2 - iter 45/154 - loss 1.69157395 - samples/sec: 24.69
2020-07-02 14:44:25,998 epoch 2 - iter 60/154 - loss 1.68283709 - samples/sec: 24.31
2020-07-02 14:44:45,498 epoch 2 - iter 75/154 - loss 1.65560424 - samples/sec: 24.78
2020-07-02 14:45:07,466 epoch 2 - iter 90/154 - loss 1.64676977 - samples/sec: 21.97
2020-07-02 14:45:27,106 epoch 2 - iter 105/154 - loss 1.63899740 - samples/sec: 24.59
2020-07-02 14:45:47,150 epoch 2 - iter 120/154 - loss 1.62948714 - samples/sec: 24.08
2020-07-02 14:46:06,680 epoch 2 - iter 135/154 - loss 1.61551479 - samples/sec: 24.88
2020-07-02 14:46:25,444 epoch 2 - iter 150/154 - loss 1.59960103 - samples/sec: 25.74
2020-07-02 14:46:30,454 ----------------------------------------------------------------------------------------------------
2020-07-02 14:46:30,455 EPOCH 2 done: loss 1.5982 - lr 0.0162181
2020-07-02 14:46:53,208 DEV : loss 1.5088865756988525 - score 0.8012
2020-07-02 14:46:53,278 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 14:46:57,546 ----------------------------------------------------------------------------------------------------
2020-07-02 14:47:19,428 epoch 3 - iter 15/154 - loss 1.42914403 - samples/sec: 22.43
2020-07-02 14:47:39,206 epoch 3 - iter 30/154 - loss 1.39760986 - samples/sec: 24.41
2020-07-02 14:47:58,944 epoch 3 - iter 45/154 - loss 1.38527177 - samples/sec: 24.45
2020-07-02 14:48:17,976 epoch 3 - iter 60/154 - loss 1.37054651 - samples/sec: 25.58
2020-07-02 14:48:37,911 epoch 3 - iter 75/154 - loss 1.35359440 - samples/sec: 24.21
2020-07-02 14:48:56,977 epoch 3 - iter 90/154 - loss 1.34525723 - samples/sec: 25.52
2020-07-02 14:49:15,853 epoch 3 - iter 105/154 - loss 1.33801149 - samples/sec: 25.58
2020-07-02 14:49:36,253 epoch 3 - iter 120/154 - loss 1.33194426 - samples/sec: 23.66
2020-07-02 14:49:56,601 epoch 3 - iter 135/154 - loss 1.32245981 - samples/sec: 23.89
2020-07-02 14:50:16,942 epoch 3 - iter 150/154 - loss 1.31203588 - samples/sec: 23.73
2020-07-02 14:50:21,354 ----------------------------------------------------------------------------------------------------
2020-07-02 14:50:21,355 EPOCH 3 done: loss 1.3032 - lr 0.0162181
2020-07-02 14:50:43,683 DEV : loss 1.2037978172302246 - score 0.8544
2020-07-02 14:50:43,903 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 14:50:48,297 ----------------------------------------------------------------------------------------------------
2020-07-02 14:51:09,072 epoch 4 - iter 15/154 - loss 1.21338186 - samples/sec: 23.39
2020-07-02 14:51:28,841 epoch 4 - iter 30/154 - loss 1.22687368 - samples/sec: 24.42
2020-07-02 14:51:48,861 epoch 4 - iter 45/154 - loss 1.18060581 - samples/sec: 24.12
2020-07-02 14:52:10,066 epoch 4 - iter 60/154 - loss 1.16962456 - samples/sec: 22.76
2020-07-02 14:52:30,187 epoch 4 - iter 75/154 - loss 1.14393969 - samples/sec: 23.99
2020-07-02 14:52:50,295 epoch 4 - iter 90/154 - loss 1.13386970 - samples/sec: 24.18
2020-07-02 14:53:09,576 epoch 4 - iter 105/154 - loss 1.12137398 - samples/sec: 25.06
2020-07-02 14:53:28,453 epoch 4 - iter 120/154 - loss 1.10854916 - samples/sec: 25.60
2020-07-02 14:53:48,347 epoch 4 - iter 135/154 - loss 1.10391057 - samples/sec: 24.27
2020-07-02 14:54:08,560 epoch 4 - iter 150/154 - loss 1.09810837 - samples/sec: 24.06
2020-07-02 14:54:14,154 ----------------------------------------------------------------------------------------------------
2020-07-02 14:54:14,156 EPOCH 4 done: loss 1.0947 - lr 0.0162181
2020-07-02 14:54:36,541 DEV : loss 0.9538484215736389 - score 0.8979
2020-07-02 14:54:36,611 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 14:54:40,893 ----------------------------------------------------------------------------------------------------
2020-07-02 14:55:03,151 epoch 5 - iter 15/154 - loss 1.02293763 - samples/sec: 22.02
2020-07-02 14:55:24,535 epoch 5 - iter 30/154 - loss 1.01685485 - samples/sec: 22.57
2020-07-02 14:55:44,082 epoch 5 - iter 45/154 - loss 1.00741227 - samples/sec: 24.70
2020-07-02 14:56:04,053 epoch 5 - iter 60/154 - loss 0.99412741 - samples/sec: 24.35
2020-07-02 14:56:24,697 epoch 5 - iter 75/154 - loss 0.97703705 - samples/sec: 23.38
2020-07-02 14:56:44,511 epoch 5 - iter 90/154 - loss 0.95509407 - samples/sec: 24.36
2020-07-02 14:57:04,272 epoch 5 - iter 105/154 - loss 0.95036031 - samples/sec: 24.61
2020-07-02 14:57:23,543 epoch 5 - iter 120/154 - loss 0.94678519 - samples/sec: 25.06
2020-07-02 14:57:42,375 epoch 5 - iter 135/154 - loss 0.93587750 - samples/sec: 25.64
2020-07-02 14:58:01,328 epoch 5 - iter 150/154 - loss 0.93406403 - samples/sec: 25.66
2020-07-02 14:58:05,957 ----------------------------------------------------------------------------------------------------
2020-07-02 14:58:05,959 EPOCH 5 done: loss 0.9277 - lr 0.0162181
2020-07-02 14:58:28,230 DEV : loss 0.8651217818260193 - score 0.8972
2020-07-02 14:58:28,297 BAD EPOCHS (no improvement): 1
2020-07-02 14:58:28,299 ----------------------------------------------------------------------------------------------------
2020-07-02 14:58:50,917 epoch 6 - iter 15/154 - loss 0.87874780 - samples/sec: 21.69
2020-07-02 14:59:10,424 epoch 6 - iter 30/154 - loss 0.84258909 - samples/sec: 24.75
2020-07-02 14:59:30,557 epoch 6 - iter 45/154 - loss 0.83004284 - samples/sec: 23.97
2020-07-02 14:59:50,051 epoch 6 - iter 60/154 - loss 0.82869032 - samples/sec: 24.99
2020-07-02 15:00:09,318 epoch 6 - iter 75/154 - loss 0.82800252 - samples/sec: 25.06
2020-07-02 15:00:28,879 epoch 6 - iter 90/154 - loss 0.82355009 - samples/sec: 24.69
2020-07-02 15:00:49,260 epoch 6 - iter 105/154 - loss 0.80982284 - samples/sec: 23.85
2020-07-02 15:01:10,265 epoch 6 - iter 120/154 - loss 0.80053075 - samples/sec: 22.98
2020-07-02 15:01:30,165 epoch 6 - iter 135/154 - loss 0.78381255 - samples/sec: 24.43
2020-07-02 15:01:49,998 epoch 6 - iter 150/154 - loss 0.77709739 - samples/sec: 24.35
2020-07-02 15:01:54,600 ----------------------------------------------------------------------------------------------------
2020-07-02 15:01:54,601 EPOCH 6 done: loss 0.7798 - lr 0.0162181
2020-07-02 15:02:16,924 DEV : loss 0.5837535262107849 - score 0.9315
2020-07-02 15:02:16,994 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 15:02:21,257 ----------------------------------------------------------------------------------------------------
2020-07-02 15:02:42,864 epoch 7 - iter 15/154 - loss 0.60823991 - samples/sec: 22.71
2020-07-02 15:03:03,054 epoch 7 - iter 30/154 - loss 0.64749694 - samples/sec: 23.92
2020-07-02 15:03:23,289 epoch 7 - iter 45/154 - loss 0.67237021 - samples/sec: 23.86
2020-07-02 15:03:44,783 epoch 7 - iter 60/154 - loss 0.67263686 - samples/sec: 22.61
2020-07-02 15:04:04,331 epoch 7 - iter 75/154 - loss 0.67410455 - samples/sec: 24.71
2020-07-02 15:04:25,483 epoch 7 - iter 90/154 - loss 0.67185280 - samples/sec: 22.82
2020-07-02 15:04:44,808 epoch 7 - iter 105/154 - loss 0.66940188 - samples/sec: 25.16
2020-07-02 15:05:03,509 epoch 7 - iter 120/154 - loss 0.67044823 - samples/sec: 25.81
2020-07-02 15:05:21,901 epoch 7 - iter 135/154 - loss 0.67026268 - samples/sec: 26.26
2020-07-02 15:05:40,996 epoch 7 - iter 150/154 - loss 0.66505048 - samples/sec: 25.45
2020-07-02 15:05:45,685 ----------------------------------------------------------------------------------------------------
2020-07-02 15:05:45,686 EPOCH 7 done: loss 0.6625 - lr 0.0162181
2020-07-02 15:06:07,975 DEV : loss 0.532750129699707 - score 0.9346
2020-07-02 15:06:08,045 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 15:06:12,337 ----------------------------------------------------------------------------------------------------
2020-07-02 15:06:32,560 epoch 8 - iter 15/154 - loss 0.61121991 - samples/sec: 24.32
2020-07-02 15:06:52,919 epoch 8 - iter 30/154 - loss 0.59228445 - samples/sec: 23.71
2020-07-02 15:07:12,810 epoch 8 - iter 45/154 - loss 0.57655472 - samples/sec: 24.28
2020-07-02 15:07:32,711 epoch 8 - iter 60/154 - loss 0.57155969 - samples/sec: 24.27
2020-07-02 15:07:52,405 epoch 8 - iter 75/154 - loss 0.56132450 - samples/sec: 24.51
2020-07-02 15:08:13,561 epoch 8 - iter 90/154 - loss 0.55938630 - samples/sec: 22.81
2020-07-02 15:08:35,125 epoch 8 - iter 105/154 - loss 0.55755164 - samples/sec: 22.51
2020-07-02 15:08:54,527 epoch 8 - iter 120/154 - loss 0.56773651 - samples/sec: 24.89
2020-07-02 15:09:14,350 epoch 8 - iter 135/154 - loss 0.56791281 - samples/sec: 24.35
2020-07-02 15:09:34,306 epoch 8 - iter 150/154 - loss 0.56556819 - samples/sec: 24.36
2020-07-02 15:09:39,031 ----------------------------------------------------------------------------------------------------
2020-07-02 15:09:39,033 EPOCH 8 done: loss 0.5619 - lr 0.0162181
2020-07-02 15:10:01,373 DEV : loss 0.6124246716499329 - score 0.9138
2020-07-02 15:10:01,443 BAD EPOCHS (no improvement): 1
2020-07-02 15:10:01,444 ----------------------------------------------------------------------------------------------------
2020-07-02 15:10:22,741 epoch 9 - iter 15/154 - loss 0.52073216 - samples/sec: 23.09
2020-07-02 15:10:41,808 epoch 9 - iter 30/154 - loss 0.51747889 - samples/sec: 25.35
2020-07-02 15:11:03,530 epoch 9 - iter 45/154 - loss 0.50766587 - samples/sec: 22.22
2020-07-02 15:11:22,831 epoch 9 - iter 60/154 - loss 0.49717654 - samples/sec: 25.19
2020-07-02 15:11:43,282 epoch 9 - iter 75/154 - loss 0.48794214 - samples/sec: 23.61
2020-07-02 15:12:03,162 epoch 9 - iter 90/154 - loss 0.48554325 - samples/sec: 24.28
2020-07-02 15:12:22,939 epoch 9 - iter 105/154 - loss 0.48294531 - samples/sec: 24.58
2020-07-02 15:12:42,470 epoch 9 - iter 120/154 - loss 0.47932543 - samples/sec: 24.72
2020-07-02 15:13:02,512 epoch 9 - iter 135/154 - loss 0.48639485 - samples/sec: 24.09
2020-07-02 15:13:22,715 epoch 9 - iter 150/154 - loss 0.48241082 - samples/sec: 24.10
2020-07-02 15:13:27,354 ----------------------------------------------------------------------------------------------------
2020-07-02 15:13:27,356 EPOCH 9 done: loss 0.4791 - lr 0.0162181
2020-07-02 15:13:49,788 DEV : loss 0.39634451270103455 - score 0.9511
2020-07-02 15:13:49,893 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 15:13:54,198 ----------------------------------------------------------------------------------------------------
2020-07-02 15:14:15,385 epoch 10 - iter 15/154 - loss 0.39700798 - samples/sec: 22.97
2020-07-02 15:14:35,579 epoch 10 - iter 30/154 - loss 0.44900593 - samples/sec: 24.16
2020-07-02 15:14:55,026 epoch 10 - iter 45/154 - loss 0.43003887 - samples/sec: 24.84
2020-07-02 15:15:14,794 epoch 10 - iter 60/154 - loss 0.44525786 - samples/sec: 24.45
2020-07-02 15:15:34,512 epoch 10 - iter 75/154 - loss 0.45256295 - samples/sec: 24.49
2020-07-02 15:15:53,671 epoch 10 - iter 90/154 - loss 0.45120489 - samples/sec: 25.21
2020-07-02 15:16:15,627 epoch 10 - iter 105/154 - loss 0.44198098 - samples/sec: 22.12
2020-07-02 15:16:36,052 epoch 10 - iter 120/154 - loss 0.43900539 - samples/sec: 23.63
2020-07-02 15:16:54,831 epoch 10 - iter 135/154 - loss 0.44348369 - samples/sec: 25.71
2020-07-02 15:17:14,348 epoch 10 - iter 150/154 - loss 0.44872592 - samples/sec: 24.91
2020-07-02 15:17:19,719 ----------------------------------------------------------------------------------------------------
2020-07-02 15:17:19,721 EPOCH 10 done: loss 0.4485 - lr 0.0162181
2020-07-02 15:17:42,071 DEV : loss 0.37473350763320923 - score 0.9554
2020-07-02 15:17:42,141 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 15:17:46,432 ----------------------------------------------------------------------------------------------------
2020-07-02 15:18:07,509 epoch 11 - iter 15/154 - loss 0.42587440 - samples/sec: 23.07
2020-07-02 15:18:26,425 epoch 11 - iter 30/154 - loss 0.41586999 - samples/sec: 25.55
2020-07-02 15:18:45,758 epoch 11 - iter 45/154 - loss 0.40622304 - samples/sec: 24.98
2020-07-02 15:19:08,078 epoch 11 - iter 60/154 - loss 0.41316052 - samples/sec: 21.62
2020-07-02 15:19:27,885 epoch 11 - iter 75/154 - loss 0.42014157 - samples/sec: 24.55
2020-07-02 15:19:46,742 epoch 11 - iter 90/154 - loss 0.40332305 - samples/sec: 25.61
2020-07-02 15:20:06,936 epoch 11 - iter 105/154 - loss 0.40566851 - samples/sec: 24.06
2020-07-02 15:20:27,452 epoch 11 - iter 120/154 - loss 0.40743910 - samples/sec: 23.53
2020-07-02 15:20:47,670 epoch 11 - iter 135/154 - loss 0.40461053 - samples/sec: 23.88
2020-07-02 15:21:07,230 epoch 11 - iter 150/154 - loss 0.40773223 - samples/sec: 24.69
2020-07-02 15:21:12,274 ----------------------------------------------------------------------------------------------------
2020-07-02 15:21:12,275 EPOCH 11 done: loss 0.4066 - lr 0.0162181
2020-07-02 15:21:34,988 DEV : loss 0.37664657831192017 - score 0.9498
2020-07-02 15:21:35,056 BAD EPOCHS (no improvement): 1
2020-07-02 15:21:35,057 ----------------------------------------------------------------------------------------------------
2020-07-02 15:21:55,878 epoch 12 - iter 15/154 - loss 0.35145220 - samples/sec: 23.35
2020-07-02 15:22:15,511 epoch 12 - iter 30/154 - loss 0.35834764 - samples/sec: 24.87
2020-07-02 15:22:34,140 epoch 12 - iter 45/154 - loss 0.35214746 - samples/sec: 25.95
2020-07-02 15:22:53,836 epoch 12 - iter 60/154 - loss 0.35563465 - samples/sec: 24.53
2020-07-02 15:23:14,561 epoch 12 - iter 75/154 - loss 0.35541137 - samples/sec: 23.29
2020-07-02 15:23:35,306 epoch 12 - iter 90/154 - loss 0.35891361 - samples/sec: 23.46
2020-07-02 15:23:55,845 epoch 12 - iter 105/154 - loss 0.36141122 - samples/sec: 23.50
2020-07-02 15:24:16,115 epoch 12 - iter 120/154 - loss 0.36918869 - samples/sec: 23.83
2020-07-02 15:24:36,956 epoch 12 - iter 135/154 - loss 0.37309432 - samples/sec: 23.30
2020-07-02 15:24:55,936 epoch 12 - iter 150/154 - loss 0.37402713 - samples/sec: 25.45
2020-07-02 15:25:01,049 ----------------------------------------------------------------------------------------------------
2020-07-02 15:25:01,051 EPOCH 12 done: loss 0.3721 - lr 0.0162181
2020-07-02 15:25:24,063 DEV : loss 0.3611939251422882 - score 0.9523
2020-07-02 15:25:24,131 BAD EPOCHS (no improvement): 2
2020-07-02 15:25:24,132 ----------------------------------------------------------------------------------------------------
2020-07-02 15:25:46,995 epoch 13 - iter 15/154 - loss 0.38793649 - samples/sec: 21.25
2020-07-02 15:26:08,016 epoch 13 - iter 30/154 - loss 0.38421851 - samples/sec: 22.98
2020-07-02 15:26:28,932 epoch 13 - iter 45/154 - loss 0.36525546 - samples/sec: 23.28
2020-07-02 15:26:49,413 epoch 13 - iter 60/154 - loss 0.36084372 - samples/sec: 23.57
2020-07-02 15:27:08,492 epoch 13 - iter 75/154 - loss 0.35162010 - samples/sec: 25.47
2020-07-02 15:27:28,847 epoch 13 - iter 90/154 - loss 0.35324366 - samples/sec: 23.76
2020-07-02 15:27:48,163 epoch 13 - iter 105/154 - loss 0.35290401 - samples/sec: 25.01
2020-07-02 15:28:06,650 epoch 13 - iter 120/154 - loss 0.35728267 - samples/sec: 26.34
2020-07-02 15:28:26,045 epoch 13 - iter 135/154 - loss 0.35024357 - samples/sec: 24.89
2020-07-02 15:28:46,154 epoch 13 - iter 150/154 - loss 0.34566470 - samples/sec: 24.00
2020-07-02 15:28:50,959 ----------------------------------------------------------------------------------------------------
2020-07-02 15:28:50,960 EPOCH 13 done: loss 0.3445 - lr 0.0162181
2020-07-02 15:29:14,173 DEV : loss 0.33947789669036865 - score 0.9578
2020-07-02 15:29:14,240 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 15:29:18,531 ----------------------------------------------------------------------------------------------------
2020-07-02 15:29:40,891 epoch 14 - iter 15/154 - loss 0.38821750 - samples/sec: 21.75
2020-07-02 15:29:59,401 epoch 14 - iter 30/154 - loss 0.35584508 - samples/sec: 26.38
2020-07-02 15:30:20,485 epoch 14 - iter 45/154 - loss 0.33287027 - samples/sec: 22.90
2020-07-02 15:30:40,526 epoch 14 - iter 60/154 - loss 0.33028220 - samples/sec: 24.09
2020-07-02 15:30:59,343 epoch 14 - iter 75/154 - loss 0.33653424 - samples/sec: 25.67
2020-07-02 15:31:19,277 epoch 14 - iter 90/154 - loss 0.33223337 - samples/sec: 24.23
2020-07-02 15:31:38,786 epoch 14 - iter 105/154 - loss 0.32999784 - samples/sec: 24.76
2020-07-02 15:31:58,166 epoch 14 - iter 120/154 - loss 0.32390204 - samples/sec: 24.94
2020-07-02 15:32:18,669 epoch 14 - iter 135/154 - loss 0.31847246 - samples/sec: 23.70
2020-07-02 15:32:38,980 epoch 14 - iter 150/154 - loss 0.31818390 - samples/sec: 23.78
2020-07-02 15:32:43,782 ----------------------------------------------------------------------------------------------------
2020-07-02 15:32:43,784 EPOCH 14 done: loss 0.3160 - lr 0.0162181
2020-07-02 15:33:06,216 DEV : loss 0.33674922585487366 - score 0.9517
2020-07-02 15:33:06,286 BAD EPOCHS (no improvement): 1
2020-07-02 15:33:06,288 ----------------------------------------------------------------------------------------------------
2020-07-02 15:33:28,131 epoch 15 - iter 15/154 - loss 0.32382143 - samples/sec: 22.46
2020-07-02 15:33:47,269 epoch 15 - iter 30/154 - loss 0.32938266 - samples/sec: 25.24
2020-07-02 15:34:08,013 epoch 15 - iter 45/154 - loss 0.35580096 - samples/sec: 23.27
2020-07-02 15:34:28,451 epoch 15 - iter 60/154 - loss 0.33785067 - samples/sec: 23.79
2020-07-02 15:34:47,864 epoch 15 - iter 75/154 - loss 0.32975845 - samples/sec: 24.87
2020-07-02 15:35:08,211 epoch 15 - iter 90/154 - loss 0.31067502 - samples/sec: 23.74
2020-07-02 15:35:27,686 epoch 15 - iter 105/154 - loss 0.30810503 - samples/sec: 24.79
2020-07-02 15:35:48,510 epoch 15 - iter 120/154 - loss 0.30373972 - samples/sec: 23.17
2020-07-02 15:36:09,118 epoch 15 - iter 135/154 - loss 0.30006551 - samples/sec: 23.44
2020-07-02 15:36:28,709 epoch 15 - iter 150/154 - loss 0.29886991 - samples/sec: 24.64
2020-07-02 15:36:33,721 ----------------------------------------------------------------------------------------------------
2020-07-02 15:36:33,723 EPOCH 15 done: loss 0.3044 - lr 0.0162181
2020-07-02 15:36:56,211 DEV : loss 0.45606178045272827 - score 0.9517
2020-07-02 15:36:56,282 BAD EPOCHS (no improvement): 2
2020-07-02 15:36:56,284 ----------------------------------------------------------------------------------------------------
2020-07-02 15:37:16,899 epoch 16 - iter 15/154 - loss 0.26865751 - samples/sec: 23.84
2020-07-02 15:37:38,328 epoch 16 - iter 30/154 - loss 0.25540573 - samples/sec: 22.51
2020-07-02 15:37:57,159 epoch 16 - iter 45/154 - loss 0.26173169 - samples/sec: 25.65
2020-07-02 15:38:17,127 epoch 16 - iter 60/154 - loss 0.26441356 - samples/sec: 24.17
2020-07-02 15:38:37,460 epoch 16 - iter 75/154 - loss 0.27951374 - samples/sec: 23.75
2020-07-02 15:38:57,341 epoch 16 - iter 90/154 - loss 0.28452394 - samples/sec: 24.43
2020-07-02 15:39:17,755 epoch 16 - iter 105/154 - loss 0.28827544 - samples/sec: 23.65
2020-07-02 15:39:37,319 epoch 16 - iter 120/154 - loss 0.29456732 - samples/sec: 24.69
2020-07-02 15:39:57,241 epoch 16 - iter 135/154 - loss 0.29623859 - samples/sec: 24.40
2020-07-02 15:40:17,446 epoch 16 - iter 150/154 - loss 0.29836056 - samples/sec: 23.89
2020-07-02 15:40:22,068 ----------------------------------------------------------------------------------------------------
2020-07-02 15:40:22,069 EPOCH 16 done: loss 0.3000 - lr 0.0162181
2020-07-02 15:40:44,528 DEV : loss 0.36227965354919434 - score 0.956
2020-07-02 15:40:44,597 BAD EPOCHS (no improvement): 3
2020-07-02 15:40:44,598 ----------------------------------------------------------------------------------------------------
2020-07-02 15:41:06,680 epoch 17 - iter 15/154 - loss 0.32204509 - samples/sec: 22.22
2020-07-02 15:41:27,155 epoch 17 - iter 30/154 - loss 0.28370522 - samples/sec: 23.60
2020-07-02 15:41:47,382 epoch 17 - iter 45/154 - loss 0.28329503 - samples/sec: 23.87
2020-07-02 15:42:06,345 epoch 17 - iter 60/154 - loss 0.28177566 - samples/sec: 25.66
2020-07-02 15:42:26,573 epoch 17 - iter 75/154 - loss 0.28397036 - samples/sec: 23.86
2020-07-02 15:42:47,908 epoch 17 - iter 90/154 - loss 0.28389345 - samples/sec: 22.63
2020-07-02 15:43:08,229 epoch 17 - iter 105/154 - loss 0.27705963 - samples/sec: 23.90
2020-07-02 15:43:28,245 epoch 17 - iter 120/154 - loss 0.27206560 - samples/sec: 24.13
2020-07-02 15:43:47,708 epoch 17 - iter 135/154 - loss 0.27306891 - samples/sec: 24.80
2020-07-02 15:44:06,403 epoch 17 - iter 150/154 - loss 0.27053858 - samples/sec: 25.85
2020-07-02 15:44:10,964 ----------------------------------------------------------------------------------------------------
2020-07-02 15:44:10,965 EPOCH 17 done: loss 0.2706 - lr 0.0162181
2020-07-02 15:44:33,212 DEV : loss 0.3132312595844269 - score 0.9682
2020-07-02 15:44:33,281 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 15:44:37,552 ----------------------------------------------------------------------------------------------------
2020-07-02 15:45:00,030 epoch 18 - iter 15/154 - loss 0.25208958 - samples/sec: 21.83
2020-07-02 15:45:19,063 epoch 18 - iter 30/154 - loss 0.27032827 - samples/sec: 25.37
2020-07-02 15:45:39,871 epoch 18 - iter 45/154 - loss 0.26369591 - samples/sec: 23.35
2020-07-02 15:46:00,460 epoch 18 - iter 60/154 - loss 0.27755907 - samples/sec: 23.44
2020-07-02 15:46:21,331 epoch 18 - iter 75/154 - loss 0.26805330 - samples/sec: 23.14
2020-07-02 15:46:39,926 epoch 18 - iter 90/154 - loss 0.26553340 - samples/sec: 25.97
2020-07-02 15:47:00,419 epoch 18 - iter 105/154 - loss 0.26726629 - samples/sec: 23.71
2020-07-02 15:47:19,530 epoch 18 - iter 120/154 - loss 0.26267663 - samples/sec: 25.27
2020-07-02 15:47:38,403 epoch 18 - iter 135/154 - loss 0.26108754 - samples/sec: 25.76
2020-07-02 15:47:58,105 epoch 18 - iter 150/154 - loss 0.26345694 - samples/sec: 24.51
2020-07-02 15:48:02,615 ----------------------------------------------------------------------------------------------------
2020-07-02 15:48:02,617 EPOCH 18 done: loss 0.2690 - lr 0.0162181
2020-07-02 15:48:24,979 DEV : loss 0.24188274145126343 - score 0.9706
2020-07-02 15:48:25,049 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 15:48:29,343 ----------------------------------------------------------------------------------------------------
2020-07-02 15:48:52,980 epoch 19 - iter 15/154 - loss 0.26782284 - samples/sec: 20.73
2020-07-02 15:49:12,800 epoch 19 - iter 30/154 - loss 0.25746150 - samples/sec: 24.36
2020-07-02 15:49:32,595 epoch 19 - iter 45/154 - loss 0.25379602 - samples/sec: 24.38
2020-07-02 15:49:51,373 epoch 19 - iter 60/154 - loss 0.25453721 - samples/sec: 25.73
2020-07-02 15:50:12,023 epoch 19 - iter 75/154 - loss 0.24971705 - samples/sec: 23.39
2020-07-02 15:50:33,769 epoch 19 - iter 90/154 - loss 0.25132714 - samples/sec: 22.20
2020-07-02 15:50:52,815 epoch 19 - iter 105/154 - loss 0.24608561 - samples/sec: 25.34
2020-07-02 15:51:12,130 epoch 19 - iter 120/154 - loss 0.24540012 - samples/sec: 25.01
2020-07-02 15:51:30,127 epoch 19 - iter 135/154 - loss 0.24441875 - samples/sec: 26.82
2020-07-02 15:51:50,334 epoch 19 - iter 150/154 - loss 0.24105846 - samples/sec: 24.05
2020-07-02 15:51:55,900 ----------------------------------------------------------------------------------------------------
2020-07-02 15:51:55,902 EPOCH 19 done: loss 0.2417 - lr 0.0162181
2020-07-02 15:52:18,249 DEV : loss 0.2675461173057556 - score 0.9645
2020-07-02 15:52:18,315 BAD EPOCHS (no improvement): 1
2020-07-02 15:52:18,317 ----------------------------------------------------------------------------------------------------
2020-07-02 15:52:40,145 epoch 20 - iter 15/154 - loss 0.22299275 - samples/sec: 22.48
2020-07-02 15:52:59,619 epoch 20 - iter 30/154 - loss 0.23901086 - samples/sec: 24.79
2020-07-02 15:53:20,568 epoch 20 - iter 45/154 - loss 0.22591903 - samples/sec: 23.04
2020-07-02 15:53:40,829 epoch 20 - iter 60/154 - loss 0.21634991 - samples/sec: 23.98
2020-07-02 15:54:01,041 epoch 20 - iter 75/154 - loss 0.22712179 - samples/sec: 23.89
2020-07-02 15:54:21,699 epoch 20 - iter 90/154 - loss 0.22611159 - samples/sec: 23.37
2020-07-02 15:54:41,296 epoch 20 - iter 105/154 - loss 0.22637414 - samples/sec: 24.81
2020-07-02 15:55:01,411 epoch 20 - iter 120/154 - loss 0.22813549 - samples/sec: 24.00
2020-07-02 15:55:21,366 epoch 20 - iter 135/154 - loss 0.22931698 - samples/sec: 24.20
2020-07-02 15:55:41,199 epoch 20 - iter 150/154 - loss 0.23357049 - samples/sec: 24.54
2020-07-02 15:55:45,334 ----------------------------------------------------------------------------------------------------
2020-07-02 15:55:45,335 EPOCH 20 done: loss 0.2297 - lr 0.0162181
2020-07-02 15:56:07,628 DEV : loss 0.23396362364292145 - score 0.9719
2020-07-02 15:56:07,700 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 15:56:11,976 ----------------------------------------------------------------------------------------------------
2020-07-02 15:56:33,755 epoch 21 - iter 15/154 - loss 0.24369793 - samples/sec: 22.53
2020-07-02 15:56:53,590 epoch 21 - iter 30/154 - loss 0.22235053 - samples/sec: 24.33
2020-07-02 15:57:13,517 epoch 21 - iter 45/154 - loss 0.22550796 - samples/sec: 24.23
2020-07-02 15:57:33,431 epoch 21 - iter 60/154 - loss 0.22605099 - samples/sec: 24.22
2020-07-02 15:57:53,093 epoch 21 - iter 75/154 - loss 0.21876556 - samples/sec: 24.57
2020-07-02 15:58:12,986 epoch 21 - iter 90/154 - loss 0.22181167 - samples/sec: 24.28
2020-07-02 15:58:33,326 epoch 21 - iter 105/154 - loss 0.22137965 - samples/sec: 23.75
2020-07-02 15:58:53,510 epoch 21 - iter 120/154 - loss 0.22005988 - samples/sec: 24.07
2020-07-02 15:59:14,691 epoch 21 - iter 135/154 - loss 0.22334215 - samples/sec: 22.80
2020-07-02 15:59:33,247 epoch 21 - iter 150/154 - loss 0.22272228 - samples/sec: 26.02
2020-07-02 15:59:38,154 ----------------------------------------------------------------------------------------------------
2020-07-02 15:59:38,156 EPOCH 21 done: loss 0.2209 - lr 0.0162181
2020-07-02 16:00:00,536 DEV : loss 0.2306308150291443 - score 0.9737
2020-07-02 16:00:00,619 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 16:00:04,925 ----------------------------------------------------------------------------------------------------
2020-07-02 16:00:27,080 epoch 22 - iter 15/154 - loss 0.15147622 - samples/sec: 21.95
2020-07-02 16:00:48,794 epoch 22 - iter 30/154 - loss 0.23374797 - samples/sec: 22.23
2020-07-02 16:01:07,535 epoch 22 - iter 45/154 - loss 0.23072744 - samples/sec: 25.76
2020-07-02 16:01:28,454 epoch 22 - iter 60/154 - loss 0.23073106 - samples/sec: 23.22
2020-07-02 16:01:47,677 epoch 22 - iter 75/154 - loss 0.22344845 - samples/sec: 25.13
2020-07-02 16:02:06,080 epoch 22 - iter 90/154 - loss 0.21383200 - samples/sec: 26.24
2020-07-02 16:02:25,456 epoch 22 - iter 105/154 - loss 0.21821867 - samples/sec: 25.10
2020-07-02 16:02:46,316 epoch 22 - iter 120/154 - loss 0.21940228 - samples/sec: 23.14
2020-07-02 16:03:06,127 epoch 22 - iter 135/154 - loss 0.21804290 - samples/sec: 24.56
2020-07-02 16:03:25,057 epoch 22 - iter 150/154 - loss 0.21798164 - samples/sec: 25.51
2020-07-02 16:03:29,479 ----------------------------------------------------------------------------------------------------
2020-07-02 16:03:29,481 EPOCH 22 done: loss 0.2170 - lr 0.0162181
2020-07-02 16:03:51,825 DEV : loss 0.2819221615791321 - score 0.9651
2020-07-02 16:03:51,897 BAD EPOCHS (no improvement): 1
2020-07-02 16:03:51,900 ----------------------------------------------------------------------------------------------------
2020-07-02 16:04:14,615 epoch 23 - iter 15/154 - loss 0.20381826 - samples/sec: 21.38
2020-07-02 16:04:36,216 epoch 23 - iter 30/154 - loss 0.21659263 - samples/sec: 22.34
2020-07-02 16:04:55,141 epoch 23 - iter 45/154 - loss 0.20468760 - samples/sec: 25.52
2020-07-02 16:05:14,993 epoch 23 - iter 60/154 - loss 0.20271637 - samples/sec: 24.48
2020-07-02 16:05:34,130 epoch 23 - iter 75/154 - loss 0.19821025 - samples/sec: 25.23
2020-07-02 16:05:54,292 epoch 23 - iter 90/154 - loss 0.20070277 - samples/sec: 24.09
2020-07-02 16:06:14,437 epoch 23 - iter 105/154 - loss 0.20031097 - samples/sec: 23.97
2020-07-02 16:06:34,083 epoch 23 - iter 120/154 - loss 0.20813754 - samples/sec: 24.58
2020-07-02 16:06:52,903 epoch 23 - iter 135/154 - loss 0.21487906 - samples/sec: 25.88
2020-07-02 16:07:13,310 epoch 23 - iter 150/154 - loss 0.20735793 - samples/sec: 23.65
2020-07-02 16:07:17,500 ----------------------------------------------------------------------------------------------------
2020-07-02 16:07:17,501 EPOCH 23 done: loss 0.2128 - lr 0.0162181
2020-07-02 16:07:39,965 DEV : loss 0.23975355923175812 - score 0.9719
2020-07-02 16:07:40,036 BAD EPOCHS (no improvement): 2
2020-07-02 16:07:40,037 ----------------------------------------------------------------------------------------------------
2020-07-02 16:08:02,215 epoch 24 - iter 15/154 - loss 0.24556303 - samples/sec: 22.19
2020-07-02 16:08:21,804 epoch 24 - iter 30/154 - loss 0.21485928 - samples/sec: 24.66
2020-07-02 16:08:42,513 epoch 24 - iter 45/154 - loss 0.20294512 - samples/sec: 23.31
2020-07-02 16:09:02,210 epoch 24 - iter 60/154 - loss 0.20490214 - samples/sec: 24.55
2020-07-02 16:09:23,391 epoch 24 - iter 75/154 - loss 0.21047717 - samples/sec: 22.93
2020-07-02 16:09:43,197 epoch 24 - iter 90/154 - loss 0.20396163 - samples/sec: 24.38
2020-07-02 16:10:04,245 epoch 24 - iter 105/154 - loss 0.20865799 - samples/sec: 22.94
2020-07-02 16:10:22,417 epoch 24 - iter 120/154 - loss 0.21269711 - samples/sec: 26.59
2020-07-02 16:10:41,693 epoch 24 - iter 135/154 - loss 0.20936885 - samples/sec: 25.06
2020-07-02 16:11:01,235 epoch 24 - iter 150/154 - loss 0.21083501 - samples/sec: 24.71
2020-07-02 16:11:05,323 ----------------------------------------------------------------------------------------------------
2020-07-02 16:11:05,324 EPOCH 24 done: loss 0.2130 - lr 0.0162181
2020-07-02 16:11:27,865 DEV : loss 0.2661949396133423 - score 0.9737
2020-07-02 16:11:27,934 BAD EPOCHS (no improvement): 3
2020-07-02 16:11:27,936 ----------------------------------------------------------------------------------------------------
2020-07-02 16:11:50,567 epoch 25 - iter 15/154 - loss 0.24687227 - samples/sec: 21.46
2020-07-02 16:12:08,728 epoch 25 - iter 30/154 - loss 0.21076790 - samples/sec: 26.60
2020-07-02 16:12:29,997 epoch 25 - iter 45/154 - loss 0.18616831 - samples/sec: 22.90
2020-07-02 16:12:48,632 epoch 25 - iter 60/154 - loss 0.18038774 - samples/sec: 25.93
2020-07-02 16:13:09,459 epoch 25 - iter 75/154 - loss 0.18513621 - samples/sec: 23.33
2020-07-02 16:13:30,101 epoch 25 - iter 90/154 - loss 0.18450534 - samples/sec: 23.39
2020-07-02 16:13:49,169 epoch 25 - iter 105/154 - loss 0.18219731 - samples/sec: 25.32
2020-07-02 16:14:09,165 epoch 25 - iter 120/154 - loss 0.18160356 - samples/sec: 24.31
2020-07-02 16:14:29,648 epoch 25 - iter 135/154 - loss 0.18727879 - samples/sec: 23.57
2020-07-02 16:14:49,182 epoch 25 - iter 150/154 - loss 0.18395177 - samples/sec: 24.71
2020-07-02 16:14:54,658 ----------------------------------------------------------------------------------------------------
2020-07-02 16:14:54,660 EPOCH 25 done: loss 0.1870 - lr 0.0162181
2020-07-02 16:15:17,417 DEV : loss 0.24831292033195496 - score 0.9725
2020-07-02 16:15:17,486 BAD EPOCHS (no improvement): 4
2020-07-02 16:15:17,487 ----------------------------------------------------------------------------------------------------
2020-07-02 16:15:40,274 epoch 26 - iter 15/154 - loss 0.15373853 - samples/sec: 21.31
2020-07-02 16:16:00,506 epoch 26 - iter 30/154 - loss 0.16712674 - samples/sec: 23.88
2020-07-02 16:16:21,136 epoch 26 - iter 45/154 - loss 0.17038985 - samples/sec: 23.63
2020-07-02 16:16:40,021 epoch 26 - iter 60/154 - loss 0.17567901 - samples/sec: 25.57
2020-07-02 16:16:59,197 epoch 26 - iter 75/154 - loss 0.18035345 - samples/sec: 25.19
2020-07-02 16:17:18,590 epoch 26 - iter 90/154 - loss 0.18694772 - samples/sec: 25.07
2020-07-02 16:17:38,407 epoch 26 - iter 105/154 - loss 0.19046586 - samples/sec: 24.37
2020-07-02 16:17:59,785 epoch 26 - iter 120/154 - loss 0.18908963 - samples/sec: 22.58
2020-07-02 16:18:19,241 epoch 26 - iter 135/154 - loss 0.18909395 - samples/sec: 24.82
2020-07-02 16:18:39,838 epoch 26 - iter 150/154 - loss 0.18379643 - samples/sec: 23.59
2020-07-02 16:18:44,586 ----------------------------------------------------------------------------------------------------
2020-07-02 16:18:44,588 EPOCH 26 done: loss 0.1831 - lr 0.0162181
2020-07-02 16:19:07,027 DEV : loss 0.23407800495624542 - score 0.9719
2020-07-02 16:19:07,096 BAD EPOCHS (no improvement): 5
2020-07-02 16:19:07,097 ----------------------------------------------------------------------------------------------------
2020-07-02 16:19:28,706 epoch 27 - iter 15/154 - loss 0.20348703 - samples/sec: 22.48
2020-07-02 16:19:48,120 epoch 27 - iter 30/154 - loss 0.16796032 - samples/sec: 24.88
2020-07-02 16:20:07,942 epoch 27 - iter 45/154 - loss 0.17932437 - samples/sec: 24.36
2020-07-02 16:20:27,121 epoch 27 - iter 60/154 - loss 0.17203238 - samples/sec: 25.18
2020-07-02 16:20:48,312 epoch 27 - iter 75/154 - loss 0.16742311 - samples/sec: 22.94
2020-07-02 16:21:08,425 epoch 27 - iter 90/154 - loss 0.17116617 - samples/sec: 24.00
2020-07-02 16:21:28,036 epoch 27 - iter 105/154 - loss 0.17072401 - samples/sec: 24.79
2020-07-02 16:21:47,868 epoch 27 - iter 120/154 - loss 0.17486551 - samples/sec: 24.34
2020-07-02 16:22:08,240 epoch 27 - iter 135/154 - loss 0.17548364 - samples/sec: 23.70
2020-07-02 16:22:28,509 epoch 27 - iter 150/154 - loss 0.17920635 - samples/sec: 24.04
2020-07-02 16:22:32,904 ----------------------------------------------------------------------------------------------------
2020-07-02 16:22:32,906 EPOCH 27 done: loss 0.1777 - lr 0.0162181
2020-07-02 16:22:55,495 DEV : loss 0.3410264253616333 - score 0.9554
Epoch    27: reducing learning rate of group 0 to 8.1091e-03.
2020-07-02 16:22:55,577 BAD EPOCHS (no improvement): 6
2020-07-02 16:22:55,578 ----------------------------------------------------------------------------------------------------
2020-07-02 16:23:16,576 epoch 28 - iter 15/154 - loss 0.18262969 - samples/sec: 23.17
2020-07-02 16:23:37,276 epoch 28 - iter 30/154 - loss 0.17605033 - samples/sec: 23.57
2020-07-02 16:23:57,641 epoch 28 - iter 45/154 - loss 0.16287563 - samples/sec: 23.70
2020-07-02 16:24:17,352 epoch 28 - iter 60/154 - loss 0.17845349 - samples/sec: 24.50
2020-07-02 16:24:37,712 epoch 28 - iter 75/154 - loss 0.16782649 - samples/sec: 23.86
2020-07-02 16:24:57,625 epoch 28 - iter 90/154 - loss 0.16389592 - samples/sec: 24.25
2020-07-02 16:25:16,025 epoch 28 - iter 105/154 - loss 0.15795042 - samples/sec: 26.24
2020-07-02 16:25:35,831 epoch 28 - iter 120/154 - loss 0.16067890 - samples/sec: 24.53
2020-07-02 16:25:54,702 epoch 28 - iter 135/154 - loss 0.16030627 - samples/sec: 25.58
2020-07-02 16:26:14,707 epoch 28 - iter 150/154 - loss 0.15914360 - samples/sec: 24.29
2020-07-02 16:26:18,949 ----------------------------------------------------------------------------------------------------
2020-07-02 16:26:18,950 EPOCH 28 done: loss 0.1580 - lr 0.0081091
2020-07-02 16:26:41,249 DEV : loss 0.2035386562347412 - score 0.9755
2020-07-02 16:26:41,318 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 16:26:45,583 ----------------------------------------------------------------------------------------------------
2020-07-02 16:27:07,154 epoch 29 - iter 15/154 - loss 0.14639383 - samples/sec: 22.52
2020-07-02 16:27:26,806 epoch 29 - iter 30/154 - loss 0.12783547 - samples/sec: 24.58
2020-07-02 16:27:48,187 epoch 29 - iter 45/154 - loss 0.13536035 - samples/sec: 22.56
2020-07-02 16:28:07,762 epoch 29 - iter 60/154 - loss 0.14223794 - samples/sec: 24.68
2020-07-02 16:28:26,724 epoch 29 - iter 75/154 - loss 0.14081989 - samples/sec: 25.65
2020-07-02 16:28:48,403 epoch 29 - iter 90/154 - loss 0.13874355 - samples/sec: 22.27
2020-07-02 16:29:09,293 epoch 29 - iter 105/154 - loss 0.14219119 - samples/sec: 23.12
2020-07-02 16:29:28,855 epoch 29 - iter 120/154 - loss 0.14479751 - samples/sec: 24.69
2020-07-02 16:29:48,408 epoch 29 - iter 135/154 - loss 0.14921473 - samples/sec: 24.70
2020-07-02 16:30:08,553 epoch 29 - iter 150/154 - loss 0.14897868 - samples/sec: 23.97
2020-07-02 16:30:12,911 ----------------------------------------------------------------------------------------------------
2020-07-02 16:30:12,913 EPOCH 29 done: loss 0.1476 - lr 0.0081091
2020-07-02 16:30:35,283 DEV : loss 0.21832378208637238 - score 0.9755
2020-07-02 16:30:35,493 BAD EPOCHS (no improvement): 1
2020-07-02 16:30:35,495 ----------------------------------------------------------------------------------------------------
2020-07-02 16:30:56,765 epoch 30 - iter 15/154 - loss 0.18098603 - samples/sec: 22.84
2020-07-02 16:31:15,369 epoch 30 - iter 30/154 - loss 0.16143890 - samples/sec: 25.97
2020-07-02 16:31:35,237 epoch 30 - iter 45/154 - loss 0.14935495 - samples/sec: 24.50
2020-07-02 16:31:55,399 epoch 30 - iter 60/154 - loss 0.14397024 - samples/sec: 23.94
2020-07-02 16:32:14,663 epoch 30 - iter 75/154 - loss 0.14854784 - samples/sec: 25.07
2020-07-02 16:32:35,534 epoch 30 - iter 90/154 - loss 0.14540661 - samples/sec: 23.14
2020-07-02 16:32:55,656 epoch 30 - iter 105/154 - loss 0.14210871 - samples/sec: 24.00
2020-07-02 16:33:15,355 epoch 30 - iter 120/154 - loss 0.14353256 - samples/sec: 24.50
2020-07-02 16:33:36,347 epoch 30 - iter 135/154 - loss 0.14563753 - samples/sec: 23.14
2020-07-02 16:33:57,331 epoch 30 - iter 150/154 - loss 0.14559265 - samples/sec: 22.99
2020-07-02 16:34:01,855 ----------------------------------------------------------------------------------------------------
2020-07-02 16:34:01,856 EPOCH 30 done: loss 0.1459 - lr 0.0081091
2020-07-02 16:34:24,512 DEV : loss 0.24465428292751312 - score 0.9719
2020-07-02 16:34:24,579 BAD EPOCHS (no improvement): 2
2020-07-02 16:34:24,580 ----------------------------------------------------------------------------------------------------
2020-07-02 16:34:46,219 epoch 31 - iter 15/154 - loss 0.15976830 - samples/sec: 22.46
2020-07-02 16:35:04,746 epoch 31 - iter 30/154 - loss 0.13884736 - samples/sec: 26.08
2020-07-02 16:35:25,418 epoch 31 - iter 45/154 - loss 0.13285553 - samples/sec: 23.56
2020-07-02 16:35:46,085 epoch 31 - iter 60/154 - loss 0.13884976 - samples/sec: 23.36
2020-07-02 16:36:06,922 epoch 31 - iter 75/154 - loss 0.14738866 - samples/sec: 23.17
2020-07-02 16:36:26,604 epoch 31 - iter 90/154 - loss 0.14702798 - samples/sec: 24.71
2020-07-02 16:36:46,280 epoch 31 - iter 105/154 - loss 0.14182458 - samples/sec: 24.55
2020-07-02 16:37:05,761 epoch 31 - iter 120/154 - loss 0.14055530 - samples/sec: 24.78
2020-07-02 16:37:26,799 epoch 31 - iter 135/154 - loss 0.13758830 - samples/sec: 23.09
2020-07-02 16:37:46,408 epoch 31 - iter 150/154 - loss 0.13833731 - samples/sec: 24.62
2020-07-02 16:37:50,524 ----------------------------------------------------------------------------------------------------
2020-07-02 16:37:50,525 EPOCH 31 done: loss 0.1374 - lr 0.0081091
2020-07-02 16:38:12,884 DEV : loss 0.20527389645576477 - score 0.9743
2020-07-02 16:38:12,956 BAD EPOCHS (no improvement): 3
2020-07-02 16:38:12,957 ----------------------------------------------------------------------------------------------------
2020-07-02 16:38:34,427 epoch 32 - iter 15/154 - loss 0.11888420 - samples/sec: 23.19
2020-07-02 16:38:55,683 epoch 32 - iter 30/154 - loss 0.13450191 - samples/sec: 22.71
2020-07-02 16:39:14,237 epoch 32 - iter 45/154 - loss 0.13657612 - samples/sec: 26.21
2020-07-02 16:39:34,635 epoch 32 - iter 60/154 - loss 0.13090624 - samples/sec: 23.67
2020-07-02 16:39:53,688 epoch 32 - iter 75/154 - loss 0.13154434 - samples/sec: 25.34
2020-07-02 16:40:13,250 epoch 32 - iter 90/154 - loss 0.13931910 - samples/sec: 24.69
2020-07-02 16:40:35,482 epoch 32 - iter 105/154 - loss 0.14349992 - samples/sec: 21.70
2020-07-02 16:40:55,671 epoch 32 - iter 120/154 - loss 0.14045170 - samples/sec: 24.08
2020-07-02 16:41:14,900 epoch 32 - iter 135/154 - loss 0.14271810 - samples/sec: 25.12
2020-07-02 16:41:33,122 epoch 32 - iter 150/154 - loss 0.14134367 - samples/sec: 26.51
2020-07-02 16:41:37,990 ----------------------------------------------------------------------------------------------------
2020-07-02 16:41:37,992 EPOCH 32 done: loss 0.1423 - lr 0.0081091
2020-07-02 16:42:00,731 DEV : loss 0.21198561787605286 - score 0.9755
2020-07-02 16:42:00,803 BAD EPOCHS (no improvement): 4
2020-07-02 16:42:00,805 ----------------------------------------------------------------------------------------------------
2020-07-02 16:42:23,967 epoch 33 - iter 15/154 - loss 0.17858952 - samples/sec: 20.97
2020-07-02 16:42:44,312 epoch 33 - iter 30/154 - loss 0.14277964 - samples/sec: 23.75
2020-07-02 16:43:04,676 epoch 33 - iter 45/154 - loss 0.14020735 - samples/sec: 23.93
2020-07-02 16:43:25,173 epoch 33 - iter 60/154 - loss 0.13840914 - samples/sec: 23.56
2020-07-02 16:43:44,300 epoch 33 - iter 75/154 - loss 0.13643146 - samples/sec: 25.25
2020-07-02 16:44:04,640 epoch 33 - iter 90/154 - loss 0.13610004 - samples/sec: 23.90
2020-07-02 16:44:24,217 epoch 33 - iter 105/154 - loss 0.13951315 - samples/sec: 24.65
2020-07-02 16:44:44,125 epoch 33 - iter 120/154 - loss 0.13773997 - samples/sec: 24.25
2020-07-02 16:45:02,697 epoch 33 - iter 135/154 - loss 0.13444212 - samples/sec: 26.22
2020-07-02 16:45:22,033 epoch 33 - iter 150/154 - loss 0.13648703 - samples/sec: 24.97
2020-07-02 16:45:26,452 ----------------------------------------------------------------------------------------------------
2020-07-02 16:45:26,454 EPOCH 33 done: loss 0.1352 - lr 0.0081091
2020-07-02 16:45:49,301 DEV : loss 0.2025325894355774 - score 0.978
2020-07-02 16:45:49,370 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 16:45:53,635 ----------------------------------------------------------------------------------------------------
2020-07-02 16:46:16,117 epoch 34 - iter 15/154 - loss 0.13611445 - samples/sec: 21.61
2020-07-02 16:46:36,415 epoch 34 - iter 30/154 - loss 0.14471161 - samples/sec: 23.79
2020-07-02 16:46:57,806 epoch 34 - iter 45/154 - loss 0.13999367 - samples/sec: 22.56
2020-07-02 16:47:17,926 epoch 34 - iter 60/154 - loss 0.13826550 - samples/sec: 24.01
2020-07-02 16:47:35,555 epoch 34 - iter 75/154 - loss 0.13903977 - samples/sec: 27.41
2020-07-02 16:47:54,789 epoch 34 - iter 90/154 - loss 0.13380337 - samples/sec: 25.11
2020-07-02 16:48:14,552 epoch 34 - iter 105/154 - loss 0.13171350 - samples/sec: 24.42
2020-07-02 16:48:34,064 epoch 34 - iter 120/154 - loss 0.13369191 - samples/sec: 24.75
2020-07-02 16:48:54,128 epoch 34 - iter 135/154 - loss 0.13411653 - samples/sec: 24.23
2020-07-02 16:49:14,657 epoch 34 - iter 150/154 - loss 0.13264349 - samples/sec: 23.51
2020-07-02 16:49:19,762 ----------------------------------------------------------------------------------------------------
2020-07-02 16:49:19,764 EPOCH 34 done: loss 0.1323 - lr 0.0081091
2020-07-02 16:49:42,706 DEV : loss 0.21952836215496063 - score 0.9761
2020-07-02 16:49:42,773 BAD EPOCHS (no improvement): 1
2020-07-02 16:49:42,775 ----------------------------------------------------------------------------------------------------
2020-07-02 16:50:04,741 epoch 35 - iter 15/154 - loss 0.14532092 - samples/sec: 22.13
2020-07-02 16:50:24,638 epoch 35 - iter 30/154 - loss 0.14132146 - samples/sec: 24.27
2020-07-02 16:50:44,619 epoch 35 - iter 45/154 - loss 0.13029910 - samples/sec: 24.36
2020-07-02 16:51:04,282 epoch 35 - iter 60/154 - loss 0.13496286 - samples/sec: 24.56
2020-07-02 16:51:25,266 epoch 35 - iter 75/154 - loss 0.12704580 - samples/sec: 22.99
2020-07-02 16:51:45,668 epoch 35 - iter 90/154 - loss 0.13263493 - samples/sec: 23.66
2020-07-02 16:52:04,516 epoch 35 - iter 105/154 - loss 0.13433218 - samples/sec: 25.62
2020-07-02 16:52:25,365 epoch 35 - iter 120/154 - loss 0.12975193 - samples/sec: 23.16
2020-07-02 16:52:43,700 epoch 35 - iter 135/154 - loss 0.12926159 - samples/sec: 26.36
2020-07-02 16:53:04,549 epoch 35 - iter 150/154 - loss 0.13078938 - samples/sec: 23.15
2020-07-02 16:53:09,340 ----------------------------------------------------------------------------------------------------
2020-07-02 16:53:09,341 EPOCH 35 done: loss 0.1326 - lr 0.0081091
2020-07-02 16:53:31,992 DEV : loss 0.21245123445987701 - score 0.9749
2020-07-02 16:53:32,061 BAD EPOCHS (no improvement): 2
2020-07-02 16:53:32,063 ----------------------------------------------------------------------------------------------------
2020-07-02 16:53:55,029 epoch 36 - iter 15/154 - loss 0.11573464 - samples/sec: 21.14
2020-07-02 16:54:14,451 epoch 36 - iter 30/154 - loss 0.13536664 - samples/sec: 25.16
2020-07-02 16:54:34,936 epoch 36 - iter 45/154 - loss 0.14638091 - samples/sec: 23.55
2020-07-02 16:54:54,032 epoch 36 - iter 60/154 - loss 0.14207099 - samples/sec: 25.29
2020-07-02 16:55:13,615 epoch 36 - iter 75/154 - loss 0.14031379 - samples/sec: 24.65
2020-07-02 16:55:32,985 epoch 36 - iter 90/154 - loss 0.13803298 - samples/sec: 24.92
2020-07-02 16:55:53,148 epoch 36 - iter 105/154 - loss 0.14540687 - samples/sec: 23.96
2020-07-02 16:56:12,803 epoch 36 - iter 120/154 - loss 0.14509310 - samples/sec: 24.57
2020-07-02 16:56:34,448 epoch 36 - iter 135/154 - loss 0.14601868 - samples/sec: 22.44
2020-07-02 16:56:53,871 epoch 36 - iter 150/154 - loss 0.13876525 - samples/sec: 24.84
2020-07-02 16:56:58,371 ----------------------------------------------------------------------------------------------------
2020-07-02 16:56:58,372 EPOCH 36 done: loss 0.1369 - lr 0.0081091
2020-07-02 16:57:21,055 DEV : loss 0.2004116028547287 - score 0.9774
2020-07-02 16:57:21,129 BAD EPOCHS (no improvement): 3
2020-07-02 16:57:21,132 ----------------------------------------------------------------------------------------------------
2020-07-02 16:57:43,702 epoch 37 - iter 15/154 - loss 0.15859565 - samples/sec: 21.53
2020-07-02 16:58:03,547 epoch 37 - iter 30/154 - loss 0.13821206 - samples/sec: 24.33
2020-07-02 16:58:23,015 epoch 37 - iter 45/154 - loss 0.14161196 - samples/sec: 25.03
2020-07-02 16:58:42,188 epoch 37 - iter 60/154 - loss 0.14231336 - samples/sec: 25.19
2020-07-02 16:59:01,640 epoch 37 - iter 75/154 - loss 0.14150818 - samples/sec: 24.82
2020-07-02 16:59:22,858 epoch 37 - iter 90/154 - loss 0.13713271 - samples/sec: 22.75
2020-07-02 16:59:44,526 epoch 37 - iter 105/154 - loss 0.13910145 - samples/sec: 22.28
2020-07-02 17:00:03,844 epoch 37 - iter 120/154 - loss 0.13740354 - samples/sec: 25.18
2020-07-02 17:00:23,183 epoch 37 - iter 135/154 - loss 0.13456342 - samples/sec: 24.97
2020-07-02 17:00:42,278 epoch 37 - iter 150/154 - loss 0.13265983 - samples/sec: 25.29
2020-07-02 17:00:46,742 ----------------------------------------------------------------------------------------------------
2020-07-02 17:00:46,743 EPOCH 37 done: loss 0.1338 - lr 0.0081091
2020-07-02 17:01:09,404 DEV : loss 0.22566574811935425 - score 0.978
2020-07-02 17:01:09,481 BAD EPOCHS (no improvement): 4
2020-07-02 17:01:09,482 ----------------------------------------------------------------------------------------------------
2020-07-02 17:01:31,189 epoch 38 - iter 15/154 - loss 0.11563111 - samples/sec: 22.40
2020-07-02 17:01:53,327 epoch 38 - iter 30/154 - loss 0.12145271 - samples/sec: 21.80
2020-07-02 17:02:13,138 epoch 38 - iter 45/154 - loss 0.13173554 - samples/sec: 24.61
2020-07-02 17:02:33,470 epoch 38 - iter 60/154 - loss 0.13281630 - samples/sec: 23.74
2020-07-02 17:02:51,911 epoch 38 - iter 75/154 - loss 0.13535262 - samples/sec: 26.19
2020-07-02 17:03:11,871 epoch 38 - iter 90/154 - loss 0.13950271 - samples/sec: 24.35
2020-07-02 17:03:30,440 epoch 38 - iter 105/154 - loss 0.13672152 - samples/sec: 26.02
2020-07-02 17:03:51,450 epoch 38 - iter 120/154 - loss 0.13120697 - samples/sec: 22.97
2020-07-02 17:04:11,038 epoch 38 - iter 135/154 - loss 0.13313321 - samples/sec: 24.81
2020-07-02 17:04:32,278 epoch 38 - iter 150/154 - loss 0.13108277 - samples/sec: 22.72
2020-07-02 17:04:36,919 ----------------------------------------------------------------------------------------------------
2020-07-02 17:04:36,920 EPOCH 38 done: loss 0.1316 - lr 0.0081091
2020-07-02 17:04:59,607 DEV : loss 0.19655393064022064 - score 0.978
2020-07-02 17:04:59,675 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 17:05:03,953 ----------------------------------------------------------------------------------------------------
2020-07-02 17:05:27,355 epoch 39 - iter 15/154 - loss 0.08606811 - samples/sec: 20.77
2020-07-02 17:05:47,467 epoch 39 - iter 30/154 - loss 0.09626995 - samples/sec: 24.00
2020-07-02 17:06:08,121 epoch 39 - iter 45/154 - loss 0.09990381 - samples/sec: 23.60
2020-07-02 17:06:28,420 epoch 39 - iter 60/154 - loss 0.10682175 - samples/sec: 23.77
2020-07-02 17:06:49,090 epoch 39 - iter 75/154 - loss 0.10827231 - samples/sec: 23.50
2020-07-02 17:07:07,984 epoch 39 - iter 90/154 - loss 0.11429365 - samples/sec: 25.57
2020-07-02 17:07:27,829 epoch 39 - iter 105/154 - loss 0.11071864 - samples/sec: 24.34
2020-07-02 17:07:47,350 epoch 39 - iter 120/154 - loss 0.11326766 - samples/sec: 24.92
2020-07-02 17:08:06,989 epoch 39 - iter 135/154 - loss 0.11420242 - samples/sec: 24.60
2020-07-02 17:08:24,776 epoch 39 - iter 150/154 - loss 0.11605036 - samples/sec: 27.14
2020-07-02 17:08:30,076 ----------------------------------------------------------------------------------------------------
2020-07-02 17:08:30,077 EPOCH 39 done: loss 0.1189 - lr 0.0081091
2020-07-02 17:08:52,666 DEV : loss 0.23231680691242218 - score 0.978
2020-07-02 17:08:52,733 BAD EPOCHS (no improvement): 1
2020-07-02 17:08:52,734 ----------------------------------------------------------------------------------------------------
2020-07-02 17:09:14,407 epoch 40 - iter 15/154 - loss 0.10929688 - samples/sec: 22.45
2020-07-02 17:09:35,102 epoch 40 - iter 30/154 - loss 0.12555539 - samples/sec: 23.57
2020-07-02 17:09:54,695 epoch 40 - iter 45/154 - loss 0.13723706 - samples/sec: 24.64
2020-07-02 17:10:15,206 epoch 40 - iter 60/154 - loss 0.13665709 - samples/sec: 23.55
2020-07-02 17:10:34,944 epoch 40 - iter 75/154 - loss 0.13865776 - samples/sec: 24.62
2020-07-02 17:10:54,804 epoch 40 - iter 90/154 - loss 0.13386259 - samples/sec: 24.31
2020-07-02 17:11:14,413 epoch 40 - iter 105/154 - loss 0.12282358 - samples/sec: 24.61
2020-07-02 17:11:34,419 epoch 40 - iter 120/154 - loss 0.12237809 - samples/sec: 24.30
2020-07-02 17:11:53,691 epoch 40 - iter 135/154 - loss 0.12271550 - samples/sec: 25.05
2020-07-02 17:12:14,398 epoch 40 - iter 150/154 - loss 0.12171200 - samples/sec: 23.32
2020-07-02 17:12:18,591 ----------------------------------------------------------------------------------------------------
2020-07-02 17:12:18,592 EPOCH 40 done: loss 0.1226 - lr 0.0081091
2020-07-02 17:12:41,415 DEV : loss 0.19786295294761658 - score 0.9786
2020-07-02 17:12:41,486 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 17:12:45,767 ----------------------------------------------------------------------------------------------------
2020-07-02 17:13:08,933 epoch 41 - iter 15/154 - loss 0.10291993 - samples/sec: 20.97
2020-07-02 17:13:28,070 epoch 41 - iter 30/154 - loss 0.11291885 - samples/sec: 25.51
2020-07-02 17:13:49,148 epoch 41 - iter 45/154 - loss 0.12107014 - samples/sec: 22.90
2020-07-02 17:14:07,473 epoch 41 - iter 60/154 - loss 0.12080107 - samples/sec: 26.35
2020-07-02 17:14:26,414 epoch 41 - iter 75/154 - loss 0.12976694 - samples/sec: 25.67
2020-07-02 17:14:46,749 epoch 41 - iter 90/154 - loss 0.12766570 - samples/sec: 23.75
2020-07-02 17:15:07,591 epoch 41 - iter 105/154 - loss 0.12382418 - samples/sec: 23.16
2020-07-02 17:15:27,330 epoch 41 - iter 120/154 - loss 0.12289315 - samples/sec: 24.47
2020-07-02 17:15:46,626 epoch 41 - iter 135/154 - loss 0.12151943 - samples/sec: 25.03
2020-07-02 17:16:07,237 epoch 41 - iter 150/154 - loss 0.11910739 - samples/sec: 23.41
2020-07-02 17:16:11,979 ----------------------------------------------------------------------------------------------------
2020-07-02 17:16:11,980 EPOCH 41 done: loss 0.1180 - lr 0.0081091
2020-07-02 17:16:34,749 DEV : loss 0.20476281642913818 - score 0.9786
2020-07-02 17:16:34,818 BAD EPOCHS (no improvement): 1
2020-07-02 17:16:34,819 ----------------------------------------------------------------------------------------------------
2020-07-02 17:16:58,607 epoch 42 - iter 15/154 - loss 0.11515100 - samples/sec: 20.43
2020-07-02 17:17:19,701 epoch 42 - iter 30/154 - loss 0.11943454 - samples/sec: 22.89
2020-07-02 17:17:39,395 epoch 42 - iter 45/154 - loss 0.11384823 - samples/sec: 24.79
2020-07-02 17:17:58,643 epoch 42 - iter 60/154 - loss 0.11443645 - samples/sec: 25.09
2020-07-02 17:18:18,648 epoch 42 - iter 75/154 - loss 0.11680269 - samples/sec: 24.13
2020-07-02 17:18:38,267 epoch 42 - iter 90/154 - loss 0.12064875 - samples/sec: 24.82
2020-07-02 17:18:58,371 epoch 42 - iter 105/154 - loss 0.12379217 - samples/sec: 24.01
2020-07-02 17:19:19,252 epoch 42 - iter 120/154 - loss 0.12142454 - samples/sec: 23.12
2020-07-02 17:19:37,911 epoch 42 - iter 135/154 - loss 0.12138695 - samples/sec: 25.90
2020-07-02 17:19:57,683 epoch 42 - iter 150/154 - loss 0.12124144 - samples/sec: 24.42
2020-07-02 17:20:02,818 ----------------------------------------------------------------------------------------------------
2020-07-02 17:20:02,819 EPOCH 42 done: loss 0.1232 - lr 0.0081091
2020-07-02 17:20:25,643 DEV : loss 0.228939026594162 - score 0.9749
2020-07-02 17:20:25,711 BAD EPOCHS (no improvement): 2
2020-07-02 17:20:25,712 ----------------------------------------------------------------------------------------------------
2020-07-02 17:20:50,520 epoch 43 - iter 15/154 - loss 0.11688481 - samples/sec: 19.56
2020-07-02 17:21:09,983 epoch 43 - iter 30/154 - loss 0.11299946 - samples/sec: 24.82
2020-07-02 17:21:31,073 epoch 43 - iter 45/154 - loss 0.10471270 - samples/sec: 23.10
2020-07-02 17:21:50,293 epoch 43 - iter 60/154 - loss 0.11333607 - samples/sec: 25.13
2020-07-02 17:22:09,299 epoch 43 - iter 75/154 - loss 0.10945214 - samples/sec: 25.40
2020-07-02 17:22:29,846 epoch 43 - iter 90/154 - loss 0.10823468 - samples/sec: 23.49
2020-07-02 17:22:48,260 epoch 43 - iter 105/154 - loss 0.11234212 - samples/sec: 26.24
2020-07-02 17:23:08,351 epoch 43 - iter 120/154 - loss 0.11211128 - samples/sec: 24.03
2020-07-02 17:23:28,918 epoch 43 - iter 135/154 - loss 0.11339033 - samples/sec: 23.64
2020-07-02 17:23:48,849 epoch 43 - iter 150/154 - loss 0.11390336 - samples/sec: 24.22
2020-07-02 17:23:53,484 ----------------------------------------------------------------------------------------------------
2020-07-02 17:23:53,486 EPOCH 43 done: loss 0.1160 - lr 0.0081091
2020-07-02 17:24:16,355 DEV : loss 0.20396070182323456 - score 0.978
2020-07-02 17:24:16,423 BAD EPOCHS (no improvement): 3
2020-07-02 17:24:16,424 ----------------------------------------------------------------------------------------------------
2020-07-02 17:24:39,845 epoch 44 - iter 15/154 - loss 0.07580647 - samples/sec: 20.76
2020-07-02 17:24:58,766 epoch 44 - iter 30/154 - loss 0.08805038 - samples/sec: 25.54
2020-07-02 17:25:19,081 epoch 44 - iter 45/154 - loss 0.08704735 - samples/sec: 23.96
2020-07-02 17:25:38,923 epoch 44 - iter 60/154 - loss 0.10597141 - samples/sec: 24.35
2020-07-02 17:25:57,992 epoch 44 - iter 75/154 - loss 0.10900353 - samples/sec: 25.31
2020-07-02 17:26:18,083 epoch 44 - iter 90/154 - loss 0.11724408 - samples/sec: 24.22
2020-07-02 17:26:36,721 epoch 44 - iter 105/154 - loss 0.11579082 - samples/sec: 25.90
2020-07-02 17:26:57,300 epoch 44 - iter 120/154 - loss 0.11354713 - samples/sec: 23.63
2020-07-02 17:27:17,545 epoch 44 - iter 135/154 - loss 0.11238661 - samples/sec: 23.84
2020-07-02 17:27:37,533 epoch 44 - iter 150/154 - loss 0.11010026 - samples/sec: 24.15
2020-07-02 17:27:41,777 ----------------------------------------------------------------------------------------------------
2020-07-02 17:27:41,778 EPOCH 44 done: loss 0.1084 - lr 0.0081091
2020-07-02 17:28:04,480 DEV : loss 0.2126724123954773 - score 0.9786
2020-07-02 17:28:04,547 BAD EPOCHS (no improvement): 4
2020-07-02 17:28:04,549 ----------------------------------------------------------------------------------------------------
2020-07-02 17:28:26,022 epoch 45 - iter 15/154 - loss 0.10441768 - samples/sec: 22.64
2020-07-02 17:28:46,731 epoch 45 - iter 30/154 - loss 0.11024101 - samples/sec: 23.31
2020-07-02 17:29:07,281 epoch 45 - iter 45/154 - loss 0.11836660 - samples/sec: 23.73
2020-07-02 17:29:26,386 epoch 45 - iter 60/154 - loss 0.11498991 - samples/sec: 25.27
2020-07-02 17:29:46,118 epoch 45 - iter 75/154 - loss 0.11378261 - samples/sec: 24.47
2020-07-02 17:30:05,930 epoch 45 - iter 90/154 - loss 0.11051767 - samples/sec: 24.54
2020-07-02 17:30:24,450 epoch 45 - iter 105/154 - loss 0.11882560 - samples/sec: 26.09
2020-07-02 17:30:43,922 epoch 45 - iter 120/154 - loss 0.11922019 - samples/sec: 24.79
2020-07-02 17:31:05,345 epoch 45 - iter 135/154 - loss 0.11827121 - samples/sec: 22.69
2020-07-02 17:31:24,900 epoch 45 - iter 150/154 - loss 0.11522523 - samples/sec: 24.68
2020-07-02 17:31:29,527 ----------------------------------------------------------------------------------------------------
2020-07-02 17:31:29,529 EPOCH 45 done: loss 0.1151 - lr 0.0081091
2020-07-02 17:31:52,285 DEV : loss 0.21013282239437103 - score 0.9774
2020-07-02 17:31:52,353 BAD EPOCHS (no improvement): 5
2020-07-02 17:31:52,354 ----------------------------------------------------------------------------------------------------
2020-07-02 17:32:15,570 epoch 46 - iter 15/154 - loss 0.11027370 - samples/sec: 20.92
2020-07-02 17:32:35,288 epoch 46 - iter 30/154 - loss 0.10980776 - samples/sec: 24.50
2020-07-02 17:32:55,630 epoch 46 - iter 45/154 - loss 0.11028625 - samples/sec: 23.73
2020-07-02 17:33:16,437 epoch 46 - iter 60/154 - loss 0.11294564 - samples/sec: 23.20
2020-07-02 17:33:36,764 epoch 46 - iter 75/154 - loss 0.11088956 - samples/sec: 23.90
2020-07-02 17:33:56,137 epoch 46 - iter 90/154 - loss 0.11293679 - samples/sec: 24.93
2020-07-02 17:34:16,271 epoch 46 - iter 105/154 - loss 0.11380949 - samples/sec: 23.98
2020-07-02 17:34:36,209 epoch 46 - iter 120/154 - loss 0.11566317 - samples/sec: 24.22
2020-07-02 17:34:55,594 epoch 46 - iter 135/154 - loss 0.11391440 - samples/sec: 24.91
2020-07-02 17:35:14,821 epoch 46 - iter 150/154 - loss 0.11119683 - samples/sec: 25.29
2020-07-02 17:35:19,264 ----------------------------------------------------------------------------------------------------
2020-07-02 17:35:19,265 EPOCH 46 done: loss 0.1132 - lr 0.0081091
2020-07-02 17:35:41,738 DEV : loss 0.2473410964012146 - score 0.9768
Epoch    46: reducing learning rate of group 0 to 4.0545e-03.
2020-07-02 17:35:41,808 BAD EPOCHS (no improvement): 6
2020-07-02 17:35:41,811 ----------------------------------------------------------------------------------------------------
2020-07-02 17:36:04,483 epoch 47 - iter 15/154 - loss 0.13017201 - samples/sec: 21.44
2020-07-02 17:36:25,431 epoch 47 - iter 30/154 - loss 0.11781174 - samples/sec: 23.27
2020-07-02 17:36:44,855 epoch 47 - iter 45/154 - loss 0.10631655 - samples/sec: 24.86
2020-07-02 17:37:03,628 epoch 47 - iter 60/154 - loss 0.09923212 - samples/sec: 25.72
2020-07-02 17:37:24,523 epoch 47 - iter 75/154 - loss 0.09882059 - samples/sec: 23.12
2020-07-02 17:37:44,449 epoch 47 - iter 90/154 - loss 0.10180322 - samples/sec: 24.21
2020-07-02 17:38:04,677 epoch 47 - iter 105/154 - loss 0.10014918 - samples/sec: 23.86
2020-07-02 17:38:24,080 epoch 47 - iter 120/154 - loss 0.10079072 - samples/sec: 25.07
2020-07-02 17:38:43,268 epoch 47 - iter 135/154 - loss 0.10352145 - samples/sec: 25.17
2020-07-02 17:39:03,680 epoch 47 - iter 150/154 - loss 0.10257754 - samples/sec: 23.81
2020-07-02 17:39:08,929 ----------------------------------------------------------------------------------------------------
2020-07-02 17:39:08,930 EPOCH 47 done: loss 0.1019 - lr 0.0040545
2020-07-02 17:39:31,341 DEV : loss 0.1972641497850418 - score 0.9798
2020-07-02 17:39:31,410 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 17:39:35,701 ----------------------------------------------------------------------------------------------------
2020-07-02 17:39:56,986 epoch 48 - iter 15/154 - loss 0.09316256 - samples/sec: 22.84
2020-07-02 17:40:18,519 epoch 48 - iter 30/154 - loss 0.09207635 - samples/sec: 22.64
2020-07-02 17:40:38,501 epoch 48 - iter 45/154 - loss 0.09080028 - samples/sec: 24.16
2020-07-02 17:40:59,896 epoch 48 - iter 60/154 - loss 0.09034919 - samples/sec: 22.56
2020-07-02 17:41:21,390 epoch 48 - iter 75/154 - loss 0.09863393 - samples/sec: 22.45
2020-07-02 17:41:40,885 epoch 48 - iter 90/154 - loss 0.09842956 - samples/sec: 24.96
2020-07-02 17:41:59,314 epoch 48 - iter 105/154 - loss 0.09847930 - samples/sec: 26.23
2020-07-02 17:42:18,382 epoch 48 - iter 120/154 - loss 0.10359679 - samples/sec: 25.32
2020-07-02 17:42:36,737 epoch 48 - iter 135/154 - loss 0.10533498 - samples/sec: 26.51
2020-07-02 17:42:56,401 epoch 48 - iter 150/154 - loss 0.10285178 - samples/sec: 24.54
2020-07-02 17:43:01,894 ----------------------------------------------------------------------------------------------------
2020-07-02 17:43:01,896 EPOCH 48 done: loss 0.1029 - lr 0.0040545
2020-07-02 17:43:24,565 DEV : loss 0.20908032357692719 - score 0.9774
2020-07-02 17:43:24,634 BAD EPOCHS (no improvement): 1
2020-07-02 17:43:24,636 ----------------------------------------------------------------------------------------------------
2020-07-02 17:43:46,260 epoch 49 - iter 15/154 - loss 0.12452545 - samples/sec: 22.74
2020-07-02 17:44:06,872 epoch 49 - iter 30/154 - loss 0.11373051 - samples/sec: 23.41
2020-07-02 17:44:25,356 epoch 49 - iter 45/154 - loss 0.11127935 - samples/sec: 26.39
2020-07-02 17:44:45,722 epoch 49 - iter 60/154 - loss 0.10167709 - samples/sec: 23.70
2020-07-02 17:45:04,808 epoch 49 - iter 75/154 - loss 0.10067866 - samples/sec: 25.31
2020-07-02 17:45:25,368 epoch 49 - iter 90/154 - loss 0.10211510 - samples/sec: 23.65
2020-07-02 17:45:45,284 epoch 49 - iter 105/154 - loss 0.09828637 - samples/sec: 24.24
2020-07-02 17:46:07,360 epoch 49 - iter 120/154 - loss 0.10255460 - samples/sec: 21.85
2020-07-02 17:46:28,038 epoch 49 - iter 135/154 - loss 0.10150222 - samples/sec: 23.51
2020-07-02 17:46:47,340 epoch 49 - iter 150/154 - loss 0.10494432 - samples/sec: 25.01
2020-07-02 17:46:51,762 ----------------------------------------------------------------------------------------------------
2020-07-02 17:46:51,764 EPOCH 49 done: loss 0.1036 - lr 0.0040545
2020-07-02 17:47:14,698 DEV : loss 0.21484945714473724 - score 0.978
2020-07-02 17:47:14,767 BAD EPOCHS (no improvement): 2
2020-07-02 17:47:14,768 ----------------------------------------------------------------------------------------------------
2020-07-02 17:47:37,461 epoch 50 - iter 15/154 - loss 0.07441427 - samples/sec: 21.41
2020-07-02 17:47:57,684 epoch 50 - iter 30/154 - loss 0.09981060 - samples/sec: 23.88
2020-07-02 17:48:17,556 epoch 50 - iter 45/154 - loss 0.09582326 - samples/sec: 24.54
2020-07-02 17:48:37,096 epoch 50 - iter 60/154 - loss 0.09657856 - samples/sec: 24.71
2020-07-02 17:48:57,920 epoch 50 - iter 75/154 - loss 0.09897010 - samples/sec: 23.19
2020-07-02 17:49:17,314 epoch 50 - iter 90/154 - loss 0.09652073 - samples/sec: 25.10
2020-07-02 17:49:36,071 epoch 50 - iter 105/154 - loss 0.09323545 - samples/sec: 25.75
2020-07-02 17:49:56,835 epoch 50 - iter 120/154 - loss 0.09363949 - samples/sec: 23.24
2020-07-02 17:50:16,600 epoch 50 - iter 135/154 - loss 0.09351738 - samples/sec: 24.60
2020-07-02 17:50:37,072 epoch 50 - iter 150/154 - loss 0.09631807 - samples/sec: 23.58
2020-07-02 17:50:41,535 ----------------------------------------------------------------------------------------------------
2020-07-02 17:50:41,536 EPOCH 50 done: loss 0.0959 - lr 0.0040545
2020-07-02 17:51:04,378 DEV : loss 0.22444875538349152 - score 0.9792
2020-07-02 17:51:04,446 BAD EPOCHS (no improvement): 3
2020-07-02 17:51:04,448 ----------------------------------------------------------------------------------------------------
2020-07-02 17:51:26,503 epoch 51 - iter 15/154 - loss 0.08688969 - samples/sec: 22.04
2020-07-02 17:51:45,925 epoch 51 - iter 30/154 - loss 0.09203181 - samples/sec: 24.88
2020-07-02 17:52:06,814 epoch 51 - iter 45/154 - loss 0.10646763 - samples/sec: 23.31
2020-07-02 17:52:26,057 epoch 51 - iter 60/154 - loss 0.11192265 - samples/sec: 25.09
2020-07-02 17:52:45,627 epoch 51 - iter 75/154 - loss 0.11106363 - samples/sec: 24.69
2020-07-02 17:53:06,167 epoch 51 - iter 90/154 - loss 0.11052981 - samples/sec: 23.63
2020-07-02 17:53:26,731 epoch 51 - iter 105/154 - loss 0.10707935 - samples/sec: 23.47
2020-07-02 17:53:46,864 epoch 51 - iter 120/154 - loss 0.10218134 - samples/sec: 23.97
2020-07-02 17:54:05,931 epoch 51 - iter 135/154 - loss 0.10484334 - samples/sec: 25.52
2020-07-02 17:54:25,902 epoch 51 - iter 150/154 - loss 0.10450326 - samples/sec: 24.18
2020-07-02 17:54:30,340 ----------------------------------------------------------------------------------------------------
2020-07-02 17:54:30,341 EPOCH 51 done: loss 0.1029 - lr 0.0040545
2020-07-02 17:54:53,014 DEV : loss 0.2026711255311966 - score 0.9792
2020-07-02 17:54:53,084 BAD EPOCHS (no improvement): 4
2020-07-02 17:54:53,085 ----------------------------------------------------------------------------------------------------
2020-07-02 17:55:16,198 epoch 52 - iter 15/154 - loss 0.09530740 - samples/sec: 21.03
2020-07-02 17:55:38,136 epoch 52 - iter 30/154 - loss 0.10926712 - samples/sec: 22.01
2020-07-02 17:55:59,339 epoch 52 - iter 45/154 - loss 0.10953697 - samples/sec: 22.97
2020-07-02 17:56:18,647 epoch 52 - iter 60/154 - loss 0.11040190 - samples/sec: 25.01
2020-07-02 17:56:36,716 epoch 52 - iter 75/154 - loss 0.10400084 - samples/sec: 26.74
2020-07-02 17:56:56,401 epoch 52 - iter 90/154 - loss 0.10083394 - samples/sec: 24.69
2020-07-02 17:57:15,759 epoch 52 - iter 105/154 - loss 0.10321290 - samples/sec: 24.94
2020-07-02 17:57:35,739 epoch 52 - iter 120/154 - loss 0.09946122 - samples/sec: 24.33
2020-07-02 17:57:54,736 epoch 52 - iter 135/154 - loss 0.09763378 - samples/sec: 25.41
2020-07-02 17:58:13,939 epoch 52 - iter 150/154 - loss 0.09765992 - samples/sec: 25.15
2020-07-02 17:58:18,133 ----------------------------------------------------------------------------------------------------
2020-07-02 17:58:18,135 EPOCH 52 done: loss 0.0961 - lr 0.0040545
2020-07-02 17:58:40,895 DEV : loss 0.1978662759065628 - score 0.9792
2020-07-02 17:58:40,965 BAD EPOCHS (no improvement): 5
2020-07-02 17:58:40,968 ----------------------------------------------------------------------------------------------------
2020-07-02 17:59:03,263 epoch 53 - iter 15/154 - loss 0.10081433 - samples/sec: 21.80
2020-07-02 17:59:23,122 epoch 53 - iter 30/154 - loss 0.12010567 - samples/sec: 24.33
2020-07-02 17:59:43,761 epoch 53 - iter 45/154 - loss 0.11009379 - samples/sec: 23.41
2020-07-02 18:00:03,389 epoch 53 - iter 60/154 - loss 0.11138892 - samples/sec: 24.61
2020-07-02 18:00:22,303 epoch 53 - iter 75/154 - loss 0.10890932 - samples/sec: 25.56
2020-07-02 18:00:42,664 epoch 53 - iter 90/154 - loss 0.10678213 - samples/sec: 23.72
2020-07-02 18:01:04,707 epoch 53 - iter 105/154 - loss 0.10528137 - samples/sec: 22.06
2020-07-02 18:01:25,323 epoch 53 - iter 120/154 - loss 0.10567519 - samples/sec: 23.42
2020-07-02 18:01:43,823 epoch 53 - iter 135/154 - loss 0.10170225 - samples/sec: 26.11
2020-07-02 18:02:03,223 epoch 53 - iter 150/154 - loss 0.09834855 - samples/sec: 25.06
2020-07-02 18:02:07,660 ----------------------------------------------------------------------------------------------------
2020-07-02 18:02:07,661 EPOCH 53 done: loss 0.0981 - lr 0.0040545
2020-07-02 18:02:30,200 DEV : loss 0.20592159032821655 - score 0.9774
Epoch    53: reducing learning rate of group 0 to 2.0273e-03.
2020-07-02 18:02:30,275 BAD EPOCHS (no improvement): 6
2020-07-02 18:02:30,277 ----------------------------------------------------------------------------------------------------
2020-07-02 18:02:50,800 epoch 54 - iter 15/154 - loss 0.08718331 - samples/sec: 23.70
2020-07-02 18:03:10,995 epoch 54 - iter 30/154 - loss 0.09210472 - samples/sec: 24.14
2020-07-02 18:03:31,577 epoch 54 - iter 45/154 - loss 0.10180214 - samples/sec: 23.46
2020-07-02 18:03:51,411 epoch 54 - iter 60/154 - loss 0.09472846 - samples/sec: 24.33
2020-07-02 18:04:11,247 epoch 54 - iter 75/154 - loss 0.08983106 - samples/sec: 24.35
2020-07-02 18:04:30,488 epoch 54 - iter 90/154 - loss 0.09004916 - samples/sec: 25.09
2020-07-02 18:04:49,388 epoch 54 - iter 105/154 - loss 0.08870099 - samples/sec: 25.73
2020-07-02 18:05:09,785 epoch 54 - iter 120/154 - loss 0.08760914 - samples/sec: 23.68
2020-07-02 18:05:30,333 epoch 54 - iter 135/154 - loss 0.08641136 - samples/sec: 23.48
2020-07-02 18:05:49,789 epoch 54 - iter 150/154 - loss 0.08918157 - samples/sec: 24.99
2020-07-02 18:05:54,518 ----------------------------------------------------------------------------------------------------
2020-07-02 18:05:54,520 EPOCH 54 done: loss 0.0910 - lr 0.0020273
2020-07-02 18:06:16,972 DEV : loss 0.19774390757083893 - score 0.9798
2020-07-02 18:06:17,041 BAD EPOCHS (no improvement): 1
2020-07-02 18:06:17,042 ----------------------------------------------------------------------------------------------------
2020-07-02 18:06:39,573 epoch 55 - iter 15/154 - loss 0.08008183 - samples/sec: 21.79
2020-07-02 18:06:59,164 epoch 55 - iter 30/154 - loss 0.08933909 - samples/sec: 24.64
2020-07-02 18:07:18,806 epoch 55 - iter 45/154 - loss 0.09008065 - samples/sec: 24.58
2020-07-02 18:07:40,535 epoch 55 - iter 60/154 - loss 0.08874504 - samples/sec: 22.36
2020-07-02 18:08:00,201 epoch 55 - iter 75/154 - loss 0.08941119 - samples/sec: 24.56
2020-07-02 18:08:20,983 epoch 55 - iter 90/154 - loss 0.08812096 - samples/sec: 23.22
2020-07-02 18:08:40,361 epoch 55 - iter 105/154 - loss 0.08716224 - samples/sec: 25.10
2020-07-02 18:09:00,385 epoch 55 - iter 120/154 - loss 0.08778634 - samples/sec: 24.11
2020-07-02 18:09:19,225 epoch 55 - iter 135/154 - loss 0.08927797 - samples/sec: 25.64
2020-07-02 18:09:38,592 epoch 55 - iter 150/154 - loss 0.09444317 - samples/sec: 25.09
2020-07-02 18:09:43,560 ----------------------------------------------------------------------------------------------------
2020-07-02 18:09:43,561 EPOCH 55 done: loss 0.0934 - lr 0.0020273
2020-07-02 18:10:05,988 DEV : loss 0.19739288091659546 - score 0.978
2020-07-02 18:10:06,059 BAD EPOCHS (no improvement): 2
2020-07-02 18:10:06,061 ----------------------------------------------------------------------------------------------------
2020-07-02 18:10:28,412 epoch 56 - iter 15/154 - loss 0.08788449 - samples/sec: 21.74
2020-07-02 18:10:46,877 epoch 56 - iter 30/154 - loss 0.08437797 - samples/sec: 26.16
2020-07-02 18:11:06,048 epoch 56 - iter 45/154 - loss 0.08514273 - samples/sec: 25.19
2020-07-02 18:11:25,647 epoch 56 - iter 60/154 - loss 0.08036039 - samples/sec: 24.79
2020-07-02 18:11:45,286 epoch 56 - iter 75/154 - loss 0.08442916 - samples/sec: 24.59
2020-07-02 18:12:06,783 epoch 56 - iter 90/154 - loss 0.08954436 - samples/sec: 22.44
2020-07-02 18:12:27,649 epoch 56 - iter 105/154 - loss 0.08976238 - samples/sec: 23.30
2020-07-02 18:12:46,342 epoch 56 - iter 120/154 - loss 0.09411574 - samples/sec: 25.84
2020-07-02 18:13:05,596 epoch 56 - iter 135/154 - loss 0.09723895 - samples/sec: 25.09
2020-07-02 18:13:24,877 epoch 56 - iter 150/154 - loss 0.09649601 - samples/sec: 25.22
2020-07-02 18:13:29,371 ----------------------------------------------------------------------------------------------------
2020-07-02 18:13:29,373 EPOCH 56 done: loss 0.0952 - lr 0.0020273
2020-07-02 18:13:51,797 DEV : loss 0.20011773705482483 - score 0.9786
2020-07-02 18:13:51,866 BAD EPOCHS (no improvement): 3
2020-07-02 18:13:51,867 ----------------------------------------------------------------------------------------------------
2020-07-02 18:14:12,911 epoch 57 - iter 15/154 - loss 0.08282442 - samples/sec: 23.12
2020-07-02 18:14:32,947 epoch 57 - iter 30/154 - loss 0.09025696 - samples/sec: 24.10
2020-07-02 18:14:53,386 epoch 57 - iter 45/154 - loss 0.09394956 - samples/sec: 23.61
2020-07-02 18:15:13,913 epoch 57 - iter 60/154 - loss 0.09126146 - samples/sec: 23.66
2020-07-02 18:15:32,907 epoch 57 - iter 75/154 - loss 0.09113265 - samples/sec: 25.43
2020-07-02 18:15:52,413 epoch 57 - iter 90/154 - loss 0.08752083 - samples/sec: 24.77
2020-07-02 18:16:14,762 epoch 57 - iter 105/154 - loss 0.09105406 - samples/sec: 21.73
2020-07-02 18:16:34,872 epoch 57 - iter 120/154 - loss 0.08909690 - samples/sec: 24.00
2020-07-02 18:16:54,106 epoch 57 - iter 135/154 - loss 0.08767471 - samples/sec: 25.10
2020-07-02 18:17:13,265 epoch 57 - iter 150/154 - loss 0.08853321 - samples/sec: 25.40
2020-07-02 18:17:17,660 ----------------------------------------------------------------------------------------------------
2020-07-02 18:17:17,661 EPOCH 57 done: loss 0.0898 - lr 0.0020273
2020-07-02 18:17:40,137 DEV : loss 0.1950114369392395 - score 0.9804
2020-07-02 18:17:40,206 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 18:17:44,473 ----------------------------------------------------------------------------------------------------
2020-07-02 18:18:05,641 epoch 58 - iter 15/154 - loss 0.09854108 - samples/sec: 23.25
2020-07-02 18:18:26,107 epoch 58 - iter 30/154 - loss 0.09823707 - samples/sec: 23.59
2020-07-02 18:18:44,691 epoch 58 - iter 45/154 - loss 0.09728588 - samples/sec: 26.01
2020-07-02 18:19:05,368 epoch 58 - iter 60/154 - loss 0.09077503 - samples/sec: 23.50
2020-07-02 18:19:25,551 epoch 58 - iter 75/154 - loss 0.09195299 - samples/sec: 23.93
2020-07-02 18:19:46,168 epoch 58 - iter 90/154 - loss 0.08965277 - samples/sec: 23.41
2020-07-02 18:20:07,091 epoch 58 - iter 105/154 - loss 0.08772397 - samples/sec: 23.22
2020-07-02 18:20:25,531 epoch 58 - iter 120/154 - loss 0.08932425 - samples/sec: 26.19
2020-07-02 18:20:44,199 epoch 58 - iter 135/154 - loss 0.09028912 - samples/sec: 25.88
2020-07-02 18:21:04,364 epoch 58 - iter 150/154 - loss 0.09111484 - samples/sec: 24.11
2020-07-02 18:21:09,689 ----------------------------------------------------------------------------------------------------
2020-07-02 18:21:09,691 EPOCH 58 done: loss 0.0919 - lr 0.0020273
2020-07-02 18:21:32,360 DEV : loss 0.19827120006084442 - score 0.9798
2020-07-02 18:21:32,428 BAD EPOCHS (no improvement): 1
2020-07-02 18:21:32,429 ----------------------------------------------------------------------------------------------------
2020-07-02 18:21:53,186 epoch 59 - iter 15/154 - loss 0.07434669 - samples/sec: 23.43
2020-07-02 18:22:13,121 epoch 59 - iter 30/154 - loss 0.08210114 - samples/sec: 24.22
2020-07-02 18:22:32,288 epoch 59 - iter 45/154 - loss 0.08505506 - samples/sec: 25.19
2020-07-02 18:22:52,595 epoch 59 - iter 60/154 - loss 0.08686025 - samples/sec: 23.79
2020-07-02 18:23:11,985 epoch 59 - iter 75/154 - loss 0.08708367 - samples/sec: 25.06
2020-07-02 18:23:32,762 epoch 59 - iter 90/154 - loss 0.09318458 - samples/sec: 23.24
2020-07-02 18:23:54,143 epoch 59 - iter 105/154 - loss 0.09390901 - samples/sec: 22.56
2020-07-02 18:24:15,226 epoch 59 - iter 120/154 - loss 0.09243909 - samples/sec: 23.06
2020-07-02 18:24:34,418 epoch 59 - iter 135/154 - loss 0.09128275 - samples/sec: 25.17
2020-07-02 18:24:54,375 epoch 59 - iter 150/154 - loss 0.09075951 - samples/sec: 24.19
2020-07-02 18:24:58,631 ----------------------------------------------------------------------------------------------------
2020-07-02 18:24:58,632 EPOCH 59 done: loss 0.0899 - lr 0.0020273
2020-07-02 18:25:21,438 DEV : loss 0.2027978152036667 - score 0.9804
2020-07-02 18:25:21,509 BAD EPOCHS (no improvement): 2
2020-07-02 18:25:21,511 ----------------------------------------------------------------------------------------------------
2020-07-02 18:25:43,241 epoch 60 - iter 15/154 - loss 0.08677354 - samples/sec: 22.39
2020-07-02 18:26:02,464 epoch 60 - iter 30/154 - loss 0.08324124 - samples/sec: 25.13
2020-07-02 18:26:22,590 epoch 60 - iter 45/154 - loss 0.08803943 - samples/sec: 23.99
2020-07-02 18:26:43,159 epoch 60 - iter 60/154 - loss 0.09347135 - samples/sec: 23.47
2020-07-02 18:27:03,283 epoch 60 - iter 75/154 - loss 0.09846938 - samples/sec: 23.98
2020-07-02 18:27:23,185 epoch 60 - iter 90/154 - loss 0.09680327 - samples/sec: 24.42
2020-07-02 18:27:42,460 epoch 60 - iter 105/154 - loss 0.09433226 - samples/sec: 25.05
2020-07-02 18:28:02,370 epoch 60 - iter 120/154 - loss 0.09501102 - samples/sec: 24.25
2020-07-02 18:28:22,500 epoch 60 - iter 135/154 - loss 0.09191312 - samples/sec: 24.15
2020-07-02 18:28:43,973 epoch 60 - iter 150/154 - loss 0.09032161 - samples/sec: 22.47
2020-07-02 18:28:48,579 ----------------------------------------------------------------------------------------------------
2020-07-02 18:28:48,581 EPOCH 60 done: loss 0.0893 - lr 0.0020273
2020-07-02 18:29:11,275 DEV : loss 0.20585915446281433 - score 0.981
2020-07-02 18:29:11,343 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 18:29:15,614 ----------------------------------------------------------------------------------------------------
2020-07-02 18:29:38,156 epoch 61 - iter 15/154 - loss 0.09983344 - samples/sec: 21.55
2020-07-02 18:29:58,238 epoch 61 - iter 30/154 - loss 0.08600884 - samples/sec: 24.05
2020-07-02 18:30:17,840 epoch 61 - iter 45/154 - loss 0.08722653 - samples/sec: 24.88
2020-07-02 18:30:38,254 epoch 61 - iter 60/154 - loss 0.09154642 - samples/sec: 23.66
2020-07-02 18:30:58,140 epoch 61 - iter 75/154 - loss 0.09060006 - samples/sec: 24.27
2020-07-02 18:31:18,830 epoch 61 - iter 90/154 - loss 0.09021203 - samples/sec: 23.49
2020-07-02 18:31:39,009 epoch 61 - iter 105/154 - loss 0.09414779 - samples/sec: 23.92
2020-07-02 18:31:57,965 epoch 61 - iter 120/154 - loss 0.09225592 - samples/sec: 25.48
2020-07-02 18:32:16,358 epoch 61 - iter 135/154 - loss 0.08870085 - samples/sec: 26.26
2020-07-02 18:32:36,767 epoch 61 - iter 150/154 - loss 0.08851716 - samples/sec: 23.65
2020-07-02 18:32:41,348 ----------------------------------------------------------------------------------------------------
2020-07-02 18:32:41,350 EPOCH 61 done: loss 0.0875 - lr 0.0020273
2020-07-02 18:33:04,158 DEV : loss 0.19547177851200104 - score 0.9792
2020-07-02 18:33:04,227 BAD EPOCHS (no improvement): 1
2020-07-02 18:33:04,229 ----------------------------------------------------------------------------------------------------
2020-07-02 18:33:27,258 epoch 62 - iter 15/154 - loss 0.08993260 - samples/sec: 21.10
2020-07-02 18:33:46,125 epoch 62 - iter 30/154 - loss 0.09509450 - samples/sec: 25.61
2020-07-02 18:34:06,236 epoch 62 - iter 45/154 - loss 0.08554634 - samples/sec: 24.25
2020-07-02 18:34:26,283 epoch 62 - iter 60/154 - loss 0.08996390 - samples/sec: 24.11
2020-07-02 18:34:45,709 epoch 62 - iter 75/154 - loss 0.09672508 - samples/sec: 24.85
2020-07-02 18:35:06,660 epoch 62 - iter 90/154 - loss 0.09301235 - samples/sec: 23.18
2020-07-02 18:35:26,352 epoch 62 - iter 105/154 - loss 0.09545991 - samples/sec: 24.52
2020-07-02 18:35:46,774 epoch 62 - iter 120/154 - loss 0.09430194 - samples/sec: 23.64
2020-07-02 18:36:05,602 epoch 62 - iter 135/154 - loss 0.09519594 - samples/sec: 25.83
2020-07-02 18:36:24,706 epoch 62 - iter 150/154 - loss 0.09227619 - samples/sec: 25.27
2020-07-02 18:36:28,841 ----------------------------------------------------------------------------------------------------
2020-07-02 18:36:28,843 EPOCH 62 done: loss 0.0933 - lr 0.0020273
2020-07-02 18:36:51,561 DEV : loss 0.1974499672651291 - score 0.9798
2020-07-02 18:36:51,627 BAD EPOCHS (no improvement): 2
2020-07-02 18:36:51,628 ----------------------------------------------------------------------------------------------------
2020-07-02 18:37:13,264 epoch 63 - iter 15/154 - loss 0.08186235 - samples/sec: 22.46
2020-07-02 18:37:33,014 epoch 63 - iter 30/154 - loss 0.08035179 - samples/sec: 24.49
2020-07-02 18:37:51,954 epoch 63 - iter 45/154 - loss 0.09017727 - samples/sec: 25.72
2020-07-02 18:38:11,628 epoch 63 - iter 60/154 - loss 0.08745349 - samples/sec: 24.55
2020-07-02 18:38:31,615 epoch 63 - iter 75/154 - loss 0.08307302 - samples/sec: 24.16
2020-07-02 18:38:52,911 epoch 63 - iter 90/154 - loss 0.08419990 - samples/sec: 22.67
2020-07-02 18:39:14,286 epoch 63 - iter 105/154 - loss 0.08808264 - samples/sec: 22.57
2020-07-02 18:39:33,426 epoch 63 - iter 120/154 - loss 0.08803327 - samples/sec: 25.24
2020-07-02 18:39:53,125 epoch 63 - iter 135/154 - loss 0.08664566 - samples/sec: 24.65
2020-07-02 18:40:12,487 epoch 63 - iter 150/154 - loss 0.08714323 - samples/sec: 24.93
2020-07-02 18:40:16,662 ----------------------------------------------------------------------------------------------------
2020-07-02 18:40:16,663 EPOCH 63 done: loss 0.0865 - lr 0.0020273
2020-07-02 18:40:39,398 DEV : loss 0.2033492624759674 - score 0.9804
2020-07-02 18:40:39,466 BAD EPOCHS (no improvement): 3
2020-07-02 18:40:39,467 ----------------------------------------------------------------------------------------------------
2020-07-02 18:41:01,377 epoch 64 - iter 15/154 - loss 0.06153077 - samples/sec: 22.19
2020-07-02 18:41:21,147 epoch 64 - iter 30/154 - loss 0.07140551 - samples/sec: 24.44
2020-07-02 18:41:42,317 epoch 64 - iter 45/154 - loss 0.08376528 - samples/sec: 23.00
2020-07-02 18:42:01,321 epoch 64 - iter 60/154 - loss 0.08542068 - samples/sec: 25.43
2020-07-02 18:42:19,752 epoch 64 - iter 75/154 - loss 0.08507289 - samples/sec: 26.20
2020-07-02 18:42:39,277 epoch 64 - iter 90/154 - loss 0.08492970 - samples/sec: 24.89
2020-07-02 18:43:00,022 epoch 64 - iter 105/154 - loss 0.08645396 - samples/sec: 23.26
2020-07-02 18:43:20,185 epoch 64 - iter 120/154 - loss 0.08791573 - samples/sec: 23.95
2020-07-02 18:43:37,747 epoch 64 - iter 135/154 - loss 0.08794287 - samples/sec: 27.71
2020-07-02 18:43:57,834 epoch 64 - iter 150/154 - loss 0.09194450 - samples/sec: 24.06
2020-07-02 18:44:02,897 ----------------------------------------------------------------------------------------------------
2020-07-02 18:44:02,898 EPOCH 64 done: loss 0.0916 - lr 0.0020273
2020-07-02 18:44:25,487 DEV : loss 0.19688229262828827 - score 0.9792
2020-07-02 18:44:25,554 BAD EPOCHS (no improvement): 4
2020-07-02 18:44:25,555 ----------------------------------------------------------------------------------------------------
2020-07-02 18:44:46,224 epoch 65 - iter 15/154 - loss 0.10502815 - samples/sec: 23.54
2020-07-02 18:45:07,083 epoch 65 - iter 30/154 - loss 0.10161052 - samples/sec: 23.14
2020-07-02 18:45:27,173 epoch 65 - iter 45/154 - loss 0.10180355 - samples/sec: 24.02
2020-07-02 18:45:46,943 epoch 65 - iter 60/154 - loss 0.10438172 - samples/sec: 24.42
2020-07-02 18:46:07,489 epoch 65 - iter 75/154 - loss 0.10476604 - samples/sec: 23.64
2020-07-02 18:46:27,512 epoch 65 - iter 90/154 - loss 0.09832354 - samples/sec: 24.11
2020-07-02 18:46:46,517 epoch 65 - iter 105/154 - loss 0.09817031 - samples/sec: 25.42
2020-07-02 18:47:07,649 epoch 65 - iter 120/154 - loss 0.09402602 - samples/sec: 23.01
2020-07-02 18:47:25,938 epoch 65 - iter 135/154 - loss 0.09160565 - samples/sec: 26.42
2020-07-02 18:47:47,747 epoch 65 - iter 150/154 - loss 0.09144586 - samples/sec: 22.14
2020-07-02 18:47:52,233 ----------------------------------------------------------------------------------------------------
2020-07-02 18:47:52,234 EPOCH 65 done: loss 0.0905 - lr 0.0020273
2020-07-02 18:48:14,454 DEV : loss 0.20111960172653198 - score 0.9804
2020-07-02 18:48:14,668 BAD EPOCHS (no improvement): 5
2020-07-02 18:48:14,670 ----------------------------------------------------------------------------------------------------
2020-07-02 18:48:36,913 epoch 66 - iter 15/154 - loss 0.07612641 - samples/sec: 21.87
2020-07-02 18:48:56,241 epoch 66 - iter 30/154 - loss 0.08649506 - samples/sec: 24.99
2020-07-02 18:49:16,092 epoch 66 - iter 45/154 - loss 0.08143805 - samples/sec: 24.53
2020-07-02 18:49:36,784 epoch 66 - iter 60/154 - loss 0.08196701 - samples/sec: 23.33
2020-07-02 18:49:58,046 epoch 66 - iter 75/154 - loss 0.08096401 - samples/sec: 22.71
2020-07-02 18:50:17,695 epoch 66 - iter 90/154 - loss 0.07823467 - samples/sec: 24.58
2020-07-02 18:50:36,144 epoch 66 - iter 105/154 - loss 0.08322996 - samples/sec: 26.39
2020-07-02 18:50:55,814 epoch 66 - iter 120/154 - loss 0.08491590 - samples/sec: 24.56
2020-07-02 18:51:16,722 epoch 66 - iter 135/154 - loss 0.08727617 - samples/sec: 23.10
2020-07-02 18:51:35,800 epoch 66 - iter 150/154 - loss 0.08726892 - samples/sec: 25.32
2020-07-02 18:51:40,273 ----------------------------------------------------------------------------------------------------
2020-07-02 18:51:40,275 EPOCH 66 done: loss 0.0883 - lr 0.0020273
2020-07-02 18:52:02,781 DEV : loss 0.19793111085891724 - score 0.9798
Epoch    66: reducing learning rate of group 0 to 1.0136e-03.
2020-07-02 18:52:02,851 BAD EPOCHS (no improvement): 6
2020-07-02 18:52:02,853 ----------------------------------------------------------------------------------------------------
2020-07-02 18:52:27,895 epoch 67 - iter 15/154 - loss 0.12642477 - samples/sec: 19.38
2020-07-02 18:52:47,763 epoch 67 - iter 30/154 - loss 0.09747878 - samples/sec: 24.54
2020-07-02 18:53:07,080 epoch 67 - iter 45/154 - loss 0.08789005 - samples/sec: 25.00
2020-07-02 18:53:26,735 epoch 67 - iter 60/154 - loss 0.08494026 - samples/sec: 24.57
2020-07-02 18:53:46,315 epoch 67 - iter 75/154 - loss 0.08639383 - samples/sec: 24.67
2020-07-02 18:54:06,712 epoch 67 - iter 90/154 - loss 0.08593876 - samples/sec: 23.81
2020-07-02 18:54:25,259 epoch 67 - iter 105/154 - loss 0.08223052 - samples/sec: 26.03
2020-07-02 18:54:44,372 epoch 67 - iter 120/154 - loss 0.07795331 - samples/sec: 25.27
2020-07-02 18:55:03,642 epoch 67 - iter 135/154 - loss 0.07723948 - samples/sec: 25.22
2020-07-02 18:55:23,678 epoch 67 - iter 150/154 - loss 0.07914580 - samples/sec: 24.12
2020-07-02 18:55:28,023 ----------------------------------------------------------------------------------------------------
2020-07-02 18:55:28,024 EPOCH 67 done: loss 0.0788 - lr 0.0010136
2020-07-02 18:55:50,431 DEV : loss 0.196882426738739 - score 0.9804
2020-07-02 18:55:50,500 BAD EPOCHS (no improvement): 1
2020-07-02 18:55:50,502 ----------------------------------------------------------------------------------------------------
2020-07-02 18:56:13,151 epoch 68 - iter 15/154 - loss 0.11901739 - samples/sec: 21.44
2020-07-02 18:56:33,569 epoch 68 - iter 30/154 - loss 0.10123495 - samples/sec: 23.64
2020-07-02 18:56:53,516 epoch 68 - iter 45/154 - loss 0.10136171 - samples/sec: 24.37
2020-07-02 18:57:14,476 epoch 68 - iter 60/154 - loss 0.09211459 - samples/sec: 23.03
2020-07-02 18:57:32,709 epoch 68 - iter 75/154 - loss 0.09430514 - samples/sec: 26.50
2020-07-02 18:57:53,853 epoch 68 - iter 90/154 - loss 0.09572636 - samples/sec: 22.84
2020-07-02 18:58:13,391 epoch 68 - iter 105/154 - loss 0.09273255 - samples/sec: 24.89
2020-07-02 18:58:33,704 epoch 68 - iter 120/154 - loss 0.09239831 - samples/sec: 23.76
2020-07-02 18:58:52,488 epoch 68 - iter 135/154 - loss 0.08849469 - samples/sec: 25.92
2020-07-02 18:59:12,410 epoch 68 - iter 150/154 - loss 0.08853157 - samples/sec: 24.24
2020-07-02 18:59:16,879 ----------------------------------------------------------------------------------------------------
2020-07-02 18:59:16,880 EPOCH 68 done: loss 0.0872 - lr 0.0010136
2020-07-02 18:59:39,437 DEV : loss 0.1948608160018921 - score 0.9798
2020-07-02 18:59:39,507 BAD EPOCHS (no improvement): 2
2020-07-02 18:59:39,509 ----------------------------------------------------------------------------------------------------
2020-07-02 19:00:03,769 epoch 69 - iter 15/154 - loss 0.09427100 - samples/sec: 20.23
2020-07-02 19:00:22,554 epoch 69 - iter 30/154 - loss 0.09612075 - samples/sec: 25.72
2020-07-02 19:00:45,090 epoch 69 - iter 45/154 - loss 0.09061571 - samples/sec: 21.41
2020-07-02 19:01:05,612 epoch 69 - iter 60/154 - loss 0.08845754 - samples/sec: 23.53
2020-07-02 19:01:24,844 epoch 69 - iter 75/154 - loss 0.08787593 - samples/sec: 25.11
2020-07-02 19:01:44,256 epoch 69 - iter 90/154 - loss 0.08266684 - samples/sec: 24.87
2020-07-02 19:02:05,028 epoch 69 - iter 105/154 - loss 0.08399499 - samples/sec: 23.39
2020-07-02 19:02:24,578 epoch 69 - iter 120/154 - loss 0.08212113 - samples/sec: 24.69
2020-07-02 19:02:44,269 epoch 69 - iter 135/154 - loss 0.08719511 - samples/sec: 24.51
2020-07-02 19:03:03,110 epoch 69 - iter 150/154 - loss 0.08658361 - samples/sec: 25.65
2020-07-02 19:03:07,068 ----------------------------------------------------------------------------------------------------
2020-07-02 19:03:07,069 EPOCH 69 done: loss 0.0862 - lr 0.0010136
2020-07-02 19:03:29,523 DEV : loss 0.2046564817428589 - score 0.9792
2020-07-02 19:03:29,606 BAD EPOCHS (no improvement): 3
2020-07-02 19:03:29,608 ----------------------------------------------------------------------------------------------------
2020-07-02 19:03:53,614 epoch 70 - iter 15/154 - loss 0.07197015 - samples/sec: 20.42
2020-07-02 19:04:12,712 epoch 70 - iter 30/154 - loss 0.06957877 - samples/sec: 25.30
2020-07-02 19:04:32,036 epoch 70 - iter 45/154 - loss 0.07140003 - samples/sec: 24.97
2020-07-02 19:04:52,137 epoch 70 - iter 60/154 - loss 0.07865867 - samples/sec: 24.19
2020-07-02 19:05:13,204 epoch 70 - iter 75/154 - loss 0.08453413 - samples/sec: 22.91
2020-07-02 19:05:33,592 epoch 70 - iter 90/154 - loss 0.08186147 - samples/sec: 23.69
2020-07-02 19:05:52,073 epoch 70 - iter 105/154 - loss 0.08007018 - samples/sec: 26.35
2020-07-02 19:06:11,649 epoch 70 - iter 120/154 - loss 0.08037839 - samples/sec: 24.67
2020-07-02 19:06:30,310 epoch 70 - iter 135/154 - loss 0.07974631 - samples/sec: 25.87
2020-07-02 19:06:50,797 epoch 70 - iter 150/154 - loss 0.07986848 - samples/sec: 23.72
2020-07-02 19:06:55,307 ----------------------------------------------------------------------------------------------------
2020-07-02 19:06:55,308 EPOCH 70 done: loss 0.0803 - lr 0.0010136
2020-07-02 19:07:17,712 DEV : loss 0.20093274116516113 - score 0.9792
2020-07-02 19:07:17,781 BAD EPOCHS (no improvement): 4
2020-07-02 19:07:17,783 ----------------------------------------------------------------------------------------------------
2020-07-02 19:07:38,876 epoch 71 - iter 15/154 - loss 0.07919160 - samples/sec: 23.06
2020-07-02 19:07:58,406 epoch 71 - iter 30/154 - loss 0.08073911 - samples/sec: 25.00
2020-07-02 19:08:18,014 epoch 71 - iter 45/154 - loss 0.07823560 - samples/sec: 24.62
2020-07-02 19:08:37,811 epoch 71 - iter 60/154 - loss 0.07837923 - samples/sec: 24.41
2020-07-02 19:08:57,770 epoch 71 - iter 75/154 - loss 0.07954901 - samples/sec: 24.36
2020-07-02 19:09:17,131 epoch 71 - iter 90/154 - loss 0.08208104 - samples/sec: 24.94
2020-07-02 19:09:36,642 epoch 71 - iter 105/154 - loss 0.08020370 - samples/sec: 24.75
2020-07-02 19:09:57,211 epoch 71 - iter 120/154 - loss 0.08218516 - samples/sec: 23.62
2020-07-02 19:10:16,405 epoch 71 - iter 135/154 - loss 0.08415195 - samples/sec: 25.17
2020-07-02 19:10:37,174 epoch 71 - iter 150/154 - loss 0.08384235 - samples/sec: 23.25
2020-07-02 19:10:43,228 ----------------------------------------------------------------------------------------------------
2020-07-02 19:10:43,229 EPOCH 71 done: loss 0.0846 - lr 0.0010136
2020-07-02 19:11:06,086 DEV : loss 0.1951770782470703 - score 0.9798
2020-07-02 19:11:06,157 BAD EPOCHS (no improvement): 5
2020-07-02 19:11:06,159 ----------------------------------------------------------------------------------------------------
2020-07-02 19:11:28,401 epoch 72 - iter 15/154 - loss 0.07519316 - samples/sec: 21.84
2020-07-02 19:11:47,943 epoch 72 - iter 30/154 - loss 0.07725611 - samples/sec: 24.71
2020-07-02 19:12:08,292 epoch 72 - iter 45/154 - loss 0.08135867 - samples/sec: 23.94
2020-07-02 19:12:28,281 epoch 72 - iter 60/154 - loss 0.09061311 - samples/sec: 24.15
2020-07-02 19:12:48,774 epoch 72 - iter 75/154 - loss 0.08907134 - samples/sec: 23.56
2020-07-02 19:13:08,770 epoch 72 - iter 90/154 - loss 0.08749927 - samples/sec: 24.14
2020-07-02 19:13:27,640 epoch 72 - iter 105/154 - loss 0.08971226 - samples/sec: 25.59
2020-07-02 19:13:46,871 epoch 72 - iter 120/154 - loss 0.08774978 - samples/sec: 25.11
2020-07-02 19:14:06,861 epoch 72 - iter 135/154 - loss 0.08598488 - samples/sec: 24.31
2020-07-02 19:14:25,930 epoch 72 - iter 150/154 - loss 0.08677177 - samples/sec: 25.33
2020-07-02 19:14:30,812 ----------------------------------------------------------------------------------------------------
2020-07-02 19:14:30,813 EPOCH 72 done: loss 0.0855 - lr 0.0010136
2020-07-02 19:14:53,238 DEV : loss 0.19694292545318604 - score 0.9798
Epoch    72: reducing learning rate of group 0 to 5.0682e-04.
2020-07-02 19:14:53,307 BAD EPOCHS (no improvement): 6
2020-07-02 19:14:53,309 ----------------------------------------------------------------------------------------------------
2020-07-02 19:15:15,456 epoch 73 - iter 15/154 - loss 0.12583278 - samples/sec: 22.17
2020-07-02 19:15:36,595 epoch 73 - iter 30/154 - loss 0.10992353 - samples/sec: 22.83
2020-07-02 19:15:57,197 epoch 73 - iter 45/154 - loss 0.09559164 - samples/sec: 23.44
2020-07-02 19:16:17,290 epoch 73 - iter 60/154 - loss 0.08711064 - samples/sec: 24.03
2020-07-02 19:16:37,625 epoch 73 - iter 75/154 - loss 0.08157262 - samples/sec: 23.75
2020-07-02 19:16:57,143 epoch 73 - iter 90/154 - loss 0.08081692 - samples/sec: 24.75
2020-07-02 19:17:17,445 epoch 73 - iter 105/154 - loss 0.08004153 - samples/sec: 23.99
2020-07-02 19:17:37,636 epoch 73 - iter 120/154 - loss 0.07894342 - samples/sec: 23.90
2020-07-02 19:17:57,886 epoch 73 - iter 135/154 - loss 0.08008806 - samples/sec: 23.84
2020-07-02 19:18:16,158 epoch 73 - iter 150/154 - loss 0.08376308 - samples/sec: 26.62
2020-07-02 19:18:20,767 ----------------------------------------------------------------------------------------------------
2020-07-02 19:18:20,769 EPOCH 73 done: loss 0.0835 - lr 0.0005068
2020-07-02 19:18:43,129 DEV : loss 0.19665665924549103 - score 0.9798
2020-07-02 19:18:43,198 BAD EPOCHS (no improvement): 1
2020-07-02 19:18:43,199 ----------------------------------------------------------------------------------------------------
2020-07-02 19:19:05,401 epoch 74 - iter 15/154 - loss 0.07577754 - samples/sec: 22.09
2020-07-02 19:19:26,893 epoch 74 - iter 30/154 - loss 0.09408136 - samples/sec: 22.46
2020-07-02 19:19:47,188 epoch 74 - iter 45/154 - loss 0.08224051 - samples/sec: 23.78
2020-07-02 19:20:06,562 epoch 74 - iter 60/154 - loss 0.09164365 - samples/sec: 25.14
2020-07-02 19:20:25,430 epoch 74 - iter 75/154 - loss 0.09134524 - samples/sec: 25.58
2020-07-02 19:20:45,004 epoch 74 - iter 90/154 - loss 0.08959111 - samples/sec: 24.67
2020-07-02 19:21:03,477 epoch 74 - iter 105/154 - loss 0.08734923 - samples/sec: 26.36
2020-07-02 19:21:23,457 epoch 74 - iter 120/154 - loss 0.08620086 - samples/sec: 24.16
2020-07-02 19:21:43,214 epoch 74 - iter 135/154 - loss 0.08835269 - samples/sec: 24.61
2020-07-02 19:22:04,287 epoch 74 - iter 150/154 - loss 0.08608258 - samples/sec: 22.90
2020-07-02 19:22:08,952 ----------------------------------------------------------------------------------------------------
2020-07-02 19:22:08,954 EPOCH 74 done: loss 0.0853 - lr 0.0005068
2020-07-02 19:22:31,319 DEV : loss 0.19973701238632202 - score 0.9798
2020-07-02 19:22:31,390 BAD EPOCHS (no improvement): 2
2020-07-02 19:22:31,391 ----------------------------------------------------------------------------------------------------
2020-07-02 19:22:53,298 epoch 75 - iter 15/154 - loss 0.07400509 - samples/sec: 22.44
2020-07-02 19:23:11,752 epoch 75 - iter 30/154 - loss 0.07855947 - samples/sec: 26.18
2020-07-02 19:23:32,338 epoch 75 - iter 45/154 - loss 0.08280147 - samples/sec: 23.45
2020-07-02 19:23:51,953 epoch 75 - iter 60/154 - loss 0.08387791 - samples/sec: 24.81
2020-07-02 19:24:11,685 epoch 75 - iter 75/154 - loss 0.09056895 - samples/sec: 24.47
2020-07-02 19:24:31,575 epoch 75 - iter 90/154 - loss 0.09015049 - samples/sec: 24.28
2020-07-02 19:24:51,849 epoch 75 - iter 105/154 - loss 0.08562251 - samples/sec: 23.97
2020-07-02 19:25:11,238 epoch 75 - iter 120/154 - loss 0.08479583 - samples/sec: 24.89
2020-07-02 19:25:31,590 epoch 75 - iter 135/154 - loss 0.08538203 - samples/sec: 23.87
2020-07-02 19:25:50,109 epoch 75 - iter 150/154 - loss 0.08300129 - samples/sec: 26.08
2020-07-02 19:25:54,984 ----------------------------------------------------------------------------------------------------
2020-07-02 19:25:54,986 EPOCH 75 done: loss 0.0832 - lr 0.0005068
2020-07-02 19:26:17,461 DEV : loss 0.19750793278217316 - score 0.9804
2020-07-02 19:26:17,531 BAD EPOCHS (no improvement): 3
2020-07-02 19:26:17,533 ----------------------------------------------------------------------------------------------------
2020-07-02 19:26:40,256 epoch 76 - iter 15/154 - loss 0.10106792 - samples/sec: 21.59
2020-07-02 19:27:00,857 epoch 76 - iter 30/154 - loss 0.08253583 - samples/sec: 23.43
2020-07-02 19:27:21,793 epoch 76 - iter 45/154 - loss 0.09450863 - samples/sec: 23.21
2020-07-02 19:27:40,802 epoch 76 - iter 60/154 - loss 0.09367372 - samples/sec: 25.41
2020-07-02 19:28:00,062 epoch 76 - iter 75/154 - loss 0.09405521 - samples/sec: 25.08
2020-07-02 19:28:19,812 epoch 76 - iter 90/154 - loss 0.08686411 - samples/sec: 24.61
2020-07-02 19:28:39,040 epoch 76 - iter 105/154 - loss 0.08190449 - samples/sec: 25.14
2020-07-02 19:28:57,359 epoch 76 - iter 120/154 - loss 0.08770324 - samples/sec: 26.38
2020-07-02 19:29:17,265 epoch 76 - iter 135/154 - loss 0.09068281 - samples/sec: 24.39
2020-07-02 19:29:37,734 epoch 76 - iter 150/154 - loss 0.08872985 - samples/sec: 23.59
2020-07-02 19:29:43,601 ----------------------------------------------------------------------------------------------------
2020-07-02 19:29:43,603 EPOCH 76 done: loss 0.0894 - lr 0.0005068
2020-07-02 19:30:06,126 DEV : loss 0.1972895711660385 - score 0.9804
2020-07-02 19:30:06,337 BAD EPOCHS (no improvement): 4
2020-07-02 19:30:06,338 ----------------------------------------------------------------------------------------------------
2020-07-02 19:30:27,741 epoch 77 - iter 15/154 - loss 0.10128506 - samples/sec: 22.72
2020-07-02 19:30:47,730 epoch 77 - iter 30/154 - loss 0.10225451 - samples/sec: 24.17
2020-07-02 19:31:08,978 epoch 77 - iter 45/154 - loss 0.09445847 - samples/sec: 22.90
2020-07-02 19:31:28,184 epoch 77 - iter 60/154 - loss 0.09758655 - samples/sec: 25.15
2020-07-02 19:31:47,413 epoch 77 - iter 75/154 - loss 0.09338443 - samples/sec: 25.11
2020-07-02 19:32:06,982 epoch 77 - iter 90/154 - loss 0.08988446 - samples/sec: 24.69
2020-07-02 19:32:26,392 epoch 77 - iter 105/154 - loss 0.08855609 - samples/sec: 24.88
2020-07-02 19:32:44,943 epoch 77 - iter 120/154 - loss 0.09002902 - samples/sec: 26.22
2020-07-02 19:33:03,714 epoch 77 - iter 135/154 - loss 0.08720513 - samples/sec: 25.72
2020-07-02 19:33:25,042 epoch 77 - iter 150/154 - loss 0.08530933 - samples/sec: 22.62
2020-07-02 19:33:29,991 ----------------------------------------------------------------------------------------------------
2020-07-02 19:33:29,992 EPOCH 77 done: loss 0.0887 - lr 0.0005068
2020-07-02 19:33:53,013 DEV : loss 0.19450919330120087 - score 0.9798
2020-07-02 19:33:53,083 BAD EPOCHS (no improvement): 5
2020-07-02 19:33:53,085 ----------------------------------------------------------------------------------------------------
2020-07-02 19:34:14,455 epoch 78 - iter 15/154 - loss 0.05516994 - samples/sec: 22.75
2020-07-02 19:34:34,304 epoch 78 - iter 30/154 - loss 0.05071913 - samples/sec: 24.59
2020-07-02 19:34:54,316 epoch 78 - iter 45/154 - loss 0.06652978 - samples/sec: 24.12
2020-07-02 19:35:13,264 epoch 78 - iter 60/154 - loss 0.07676350 - samples/sec: 25.50
2020-07-02 19:35:32,371 epoch 78 - iter 75/154 - loss 0.07474116 - samples/sec: 25.28
2020-07-02 19:35:52,925 epoch 78 - iter 90/154 - loss 0.07864122 - samples/sec: 23.65
2020-07-02 19:36:12,549 epoch 78 - iter 105/154 - loss 0.08017938 - samples/sec: 24.62
2020-07-02 19:36:33,222 epoch 78 - iter 120/154 - loss 0.08172559 - samples/sec: 23.40
2020-07-02 19:36:54,373 epoch 78 - iter 135/154 - loss 0.08014307 - samples/sec: 22.96
2020-07-02 19:37:14,628 epoch 78 - iter 150/154 - loss 0.07845808 - samples/sec: 23.83
2020-07-02 19:37:19,890 ----------------------------------------------------------------------------------------------------
2020-07-02 19:37:19,891 EPOCH 78 done: loss 0.0799 - lr 0.0005068
2020-07-02 19:37:42,493 DEV : loss 0.19961152970790863 - score 0.9786
Epoch    78: reducing learning rate of group 0 to 2.5341e-04.
2020-07-02 19:37:42,593 BAD EPOCHS (no improvement): 6
2020-07-02 19:37:42,595 ----------------------------------------------------------------------------------------------------
2020-07-02 19:38:03,566 epoch 79 - iter 15/154 - loss 0.07821662 - samples/sec: 23.50
2020-07-02 19:38:22,832 epoch 79 - iter 30/154 - loss 0.08100213 - samples/sec: 25.07
2020-07-02 19:38:42,538 epoch 79 - iter 45/154 - loss 0.08145519 - samples/sec: 24.51
2020-07-02 19:39:02,437 epoch 79 - iter 60/154 - loss 0.08334038 - samples/sec: 24.25
2020-07-02 19:39:22,840 epoch 79 - iter 75/154 - loss 0.08079175 - samples/sec: 23.67
2020-07-02 19:39:44,219 epoch 79 - iter 90/154 - loss 0.08068648 - samples/sec: 22.59
2020-07-02 19:40:04,445 epoch 79 - iter 105/154 - loss 0.08182918 - samples/sec: 23.88
2020-07-02 19:40:23,599 epoch 79 - iter 120/154 - loss 0.08141914 - samples/sec: 25.42
2020-07-02 19:40:43,474 epoch 79 - iter 135/154 - loss 0.08588830 - samples/sec: 24.30
2020-07-02 19:41:02,682 epoch 79 - iter 150/154 - loss 0.08469767 - samples/sec: 25.15
2020-07-02 19:41:07,627 ----------------------------------------------------------------------------------------------------
2020-07-02 19:41:07,628 EPOCH 79 done: loss 0.0856 - lr 0.0002534
2020-07-02 19:41:30,538 DEV : loss 0.19733120501041412 - score 0.9798
2020-07-02 19:41:30,606 BAD EPOCHS (no improvement): 1
2020-07-02 19:41:30,609 ----------------------------------------------------------------------------------------------------
2020-07-02 19:41:54,035 epoch 80 - iter 15/154 - loss 0.08911656 - samples/sec: 20.74
2020-07-02 19:42:14,513 epoch 80 - iter 30/154 - loss 0.09176585 - samples/sec: 23.79
2020-07-02 19:42:33,470 epoch 80 - iter 45/154 - loss 0.08643438 - samples/sec: 25.47
2020-07-02 19:42:52,061 epoch 80 - iter 60/154 - loss 0.08399421 - samples/sec: 25.99
2020-07-02 19:43:13,561 epoch 80 - iter 75/154 - loss 0.08570782 - samples/sec: 22.60
2020-07-02 19:43:33,324 epoch 80 - iter 90/154 - loss 0.08288445 - samples/sec: 24.43
2020-07-02 19:43:52,847 epoch 80 - iter 105/154 - loss 0.08118652 - samples/sec: 24.72
2020-07-02 19:44:13,126 epoch 80 - iter 120/154 - loss 0.08139978 - samples/sec: 23.96
2020-07-02 19:44:33,738 epoch 80 - iter 135/154 - loss 0.08077319 - samples/sec: 23.42
2020-07-02 19:44:54,207 epoch 80 - iter 150/154 - loss 0.08154344 - samples/sec: 23.58
2020-07-02 19:44:58,843 ----------------------------------------------------------------------------------------------------
2020-07-02 19:44:58,845 EPOCH 80 done: loss 0.0816 - lr 0.0002534
2020-07-02 19:45:21,976 DEV : loss 0.19819292426109314 - score 0.9798
2020-07-02 19:45:22,052 BAD EPOCHS (no improvement): 2
2020-07-02 19:45:22,055 ----------------------------------------------------------------------------------------------------
2020-07-02 19:45:43,473 epoch 81 - iter 15/154 - loss 0.11282083 - samples/sec: 22.71
2020-07-02 19:46:24,008 epoch 81 - iter 45/154 - loss 0.08270191 - samples/sec: 23.23
2020-07-02 19:46:42,832 epoch 81 - iter 60/154 - loss 0.08318824 - samples/sec: 25.66
2020-07-02 19:47:02,499 epoch 81 - iter 75/154 - loss 0.08471441 - samples/sec: 24.57
2020-07-02 19:47:22,738 epoch 81 - iter 90/154 - loss 0.08327459 - samples/sec: 23.85
2020-07-02 19:47:42,972 epoch 81 - iter 105/154 - loss 0.07872031 - samples/sec: 23.86
2020-07-02 19:48:01,928 epoch 81 - iter 120/154 - loss 0.07652540 - samples/sec: 25.68
2020-07-02 19:48:22,614 epoch 81 - iter 135/154 - loss 0.07975565 - samples/sec: 23.34
2020-07-02 19:48:41,787 epoch 81 - iter 150/154 - loss 0.07985378 - samples/sec: 25.18
2020-07-02 19:48:46,539 ----------------------------------------------------------------------------------------------------
2020-07-02 19:48:46,541 EPOCH 81 done: loss 0.0798 - lr 0.0002534
2020-07-02 19:49:09,276 DEV : loss 0.19881562888622284 - score 0.9792
2020-07-02 19:49:09,344 BAD EPOCHS (no improvement): 3
2020-07-02 19:49:09,346 ----------------------------------------------------------------------------------------------------
2020-07-02 19:49:33,998 epoch 82 - iter 15/154 - loss 0.08336314 - samples/sec: 19.69
2020-07-02 19:49:53,295 epoch 82 - iter 30/154 - loss 0.08455563 - samples/sec: 25.27
2020-07-02 19:50:12,593 epoch 82 - iter 45/154 - loss 0.08075597 - samples/sec: 25.02
2020-07-02 19:50:31,678 epoch 82 - iter 60/154 - loss 0.07854668 - samples/sec: 25.30
2020-07-02 19:50:52,930 epoch 82 - iter 75/154 - loss 0.08441896 - samples/sec: 22.72
2020-07-02 19:51:12,043 epoch 82 - iter 90/154 - loss 0.07963300 - samples/sec: 25.26
2020-07-02 19:51:29,700 epoch 82 - iter 105/154 - loss 0.08339172 - samples/sec: 27.55
2020-07-02 19:51:49,879 epoch 82 - iter 120/154 - loss 0.08139258 - samples/sec: 23.93
2020-07-02 19:52:09,504 epoch 82 - iter 135/154 - loss 0.08187230 - samples/sec: 24.59
2020-07-02 19:52:29,636 epoch 82 - iter 150/154 - loss 0.08149730 - samples/sec: 24.17
2020-07-02 19:52:34,259 ----------------------------------------------------------------------------------------------------
2020-07-02 19:52:34,261 EPOCH 82 done: loss 0.0802 - lr 0.0002534
2020-07-02 19:52:56,633 DEV : loss 0.1967770755290985 - score 0.9798
2020-07-02 19:52:56,703 BAD EPOCHS (no improvement): 4
2020-07-02 19:52:56,706 ----------------------------------------------------------------------------------------------------
2020-07-02 19:53:18,491 epoch 83 - iter 15/154 - loss 0.08312181 - samples/sec: 22.32
2020-07-02 19:53:39,588 epoch 83 - iter 30/154 - loss 0.08574174 - samples/sec: 23.12
2020-07-02 19:53:58,547 epoch 83 - iter 45/154 - loss 0.08475116 - samples/sec: 25.48
2020-07-02 19:54:19,207 epoch 83 - iter 60/154 - loss 0.07566535 - samples/sec: 23.38
2020-07-02 19:54:39,070 epoch 83 - iter 75/154 - loss 0.07675807 - samples/sec: 24.31
2020-07-02 19:54:59,788 epoch 83 - iter 90/154 - loss 0.07665076 - samples/sec: 23.43
2020-07-02 19:55:19,537 epoch 83 - iter 105/154 - loss 0.07946286 - samples/sec: 24.45
2020-07-02 19:55:39,566 epoch 83 - iter 120/154 - loss 0.07992272 - samples/sec: 24.10
2020-07-02 19:55:57,919 epoch 83 - iter 135/154 - loss 0.08126198 - samples/sec: 26.51
2020-07-02 19:56:17,369 epoch 83 - iter 150/154 - loss 0.08188783 - samples/sec: 24.81
2020-07-02 19:56:21,455 ----------------------------------------------------------------------------------------------------
2020-07-02 19:56:21,456 EPOCH 83 done: loss 0.0820 - lr 0.0002534
2020-07-02 19:56:44,267 DEV : loss 0.19680184125900269 - score 0.9798
2020-07-02 19:56:44,335 BAD EPOCHS (no improvement): 5
2020-07-02 19:56:44,337 ----------------------------------------------------------------------------------------------------
2020-07-02 19:57:06,208 epoch 84 - iter 15/154 - loss 0.06290983 - samples/sec: 22.22
2020-07-02 19:57:24,662 epoch 84 - iter 30/154 - loss 0.07252258 - samples/sec: 26.19
2020-07-02 19:57:43,378 epoch 84 - iter 45/154 - loss 0.07217186 - samples/sec: 26.06
2020-07-02 19:58:03,579 epoch 84 - iter 60/154 - loss 0.07073096 - samples/sec: 23.90
2020-07-02 19:58:23,163 epoch 84 - iter 75/154 - loss 0.07663974 - samples/sec: 24.65
2020-07-02 19:58:42,104 epoch 84 - iter 90/154 - loss 0.07967136 - samples/sec: 25.69
2020-07-02 19:59:01,744 epoch 84 - iter 105/154 - loss 0.08082978 - samples/sec: 24.59
2020-07-02 19:59:23,133 epoch 84 - iter 120/154 - loss 0.08198013 - samples/sec: 22.57
2020-07-02 19:59:43,063 epoch 84 - iter 135/154 - loss 0.08208106 - samples/sec: 24.39
2020-07-02 20:00:04,913 epoch 84 - iter 150/154 - loss 0.08289737 - samples/sec: 22.07
2020-07-02 20:00:09,399 ----------------------------------------------------------------------------------------------------
2020-07-02 20:00:09,400 EPOCH 84 done: loss 0.0832 - lr 0.0002534
2020-07-02 20:00:31,750 DEV : loss 0.19807276129722595 - score 0.9798
Epoch    84: reducing learning rate of group 0 to 1.2670e-04.
2020-07-02 20:00:31,974 BAD EPOCHS (no improvement): 6
2020-07-02 20:00:31,976 ----------------------------------------------------------------------------------------------------
2020-07-02 20:00:55,472 epoch 85 - iter 15/154 - loss 0.08761623 - samples/sec: 20.67
2020-07-02 20:01:14,145 epoch 85 - iter 30/154 - loss 0.08436980 - samples/sec: 25.88
2020-07-02 20:01:34,055 epoch 85 - iter 45/154 - loss 0.08638632 - samples/sec: 24.25
2020-07-02 20:01:53,168 epoch 85 - iter 60/154 - loss 0.09292158 - samples/sec: 25.49
2020-07-02 20:02:12,416 epoch 85 - iter 75/154 - loss 0.08979836 - samples/sec: 25.09
2020-07-02 20:02:32,606 epoch 85 - iter 90/154 - loss 0.08738348 - samples/sec: 23.91
2020-07-02 20:02:53,312 epoch 85 - iter 105/154 - loss 0.08728358 - samples/sec: 23.47
2020-07-02 20:03:13,522 epoch 85 - iter 120/154 - loss 0.08704443 - samples/sec: 23.89
2020-07-02 20:03:32,671 epoch 85 - iter 135/154 - loss 0.08533383 - samples/sec: 25.23
2020-07-02 20:03:53,771 epoch 85 - iter 150/154 - loss 0.08561399 - samples/sec: 23.03
2020-07-02 20:03:58,432 ----------------------------------------------------------------------------------------------------
2020-07-02 20:03:58,433 EPOCH 85 done: loss 0.0846 - lr 0.0001267
2020-07-02 20:04:20,935 DEV : loss 0.19647707045078278 - score 0.9798
2020-07-02 20:04:21,004 BAD EPOCHS (no improvement): 1
2020-07-02 20:04:21,006 ----------------------------------------------------------------------------------------------------
2020-07-02 20:04:44,697 epoch 86 - iter 15/154 - loss 0.08141229 - samples/sec: 20.51
2020-07-02 20:05:05,999 epoch 86 - iter 30/154 - loss 0.08228939 - samples/sec: 22.67
2020-07-02 20:05:26,309 epoch 86 - iter 45/154 - loss 0.07569061 - samples/sec: 23.76
2020-07-02 20:05:45,276 epoch 86 - iter 60/154 - loss 0.07859332 - samples/sec: 25.46
2020-07-02 20:06:05,770 epoch 86 - iter 75/154 - loss 0.08294603 - samples/sec: 23.69
2020-07-02 20:06:25,472 epoch 86 - iter 90/154 - loss 0.08264545 - samples/sec: 24.51
2020-07-02 20:06:45,127 epoch 86 - iter 105/154 - loss 0.08138939 - samples/sec: 24.56
2020-07-02 20:07:03,601 epoch 86 - iter 120/154 - loss 0.08116858 - samples/sec: 26.35
2020-07-02 20:07:22,317 epoch 86 - iter 135/154 - loss 0.08432639 - samples/sec: 25.80
2020-07-02 20:07:40,946 epoch 86 - iter 150/154 - loss 0.08361728 - samples/sec: 25.93
2020-07-02 20:07:46,002 ----------------------------------------------------------------------------------------------------
2020-07-02 20:07:46,003 EPOCH 86 done: loss 0.0821 - lr 0.0001267
2020-07-02 20:08:08,553 DEV : loss 0.19691519439220428 - score 0.9798
2020-07-02 20:08:08,622 BAD EPOCHS (no improvement): 2
2020-07-02 20:08:08,623 ----------------------------------------------------------------------------------------------------
2020-07-02 20:08:32,228 epoch 87 - iter 15/154 - loss 0.07756042 - samples/sec: 20.58
2020-07-02 20:08:51,084 epoch 87 - iter 30/154 - loss 0.08246949 - samples/sec: 25.92
2020-07-02 20:09:10,734 epoch 87 - iter 45/154 - loss 0.08848242 - samples/sec: 24.57
2020-07-02 20:09:30,149 epoch 87 - iter 60/154 - loss 0.09197148 - samples/sec: 24.87
2020-07-02 20:09:52,124 epoch 87 - iter 75/154 - loss 0.09220921 - samples/sec: 22.09
2020-07-02 20:10:12,434 epoch 87 - iter 90/154 - loss 0.09423689 - samples/sec: 23.77
2020-07-02 20:10:31,990 epoch 87 - iter 105/154 - loss 0.09056646 - samples/sec: 24.69
2020-07-02 20:10:51,277 epoch 87 - iter 120/154 - loss 0.08802644 - samples/sec: 25.23
2020-07-02 20:11:11,081 epoch 87 - iter 135/154 - loss 0.08470457 - samples/sec: 24.38
2020-07-02 20:11:30,365 epoch 87 - iter 150/154 - loss 0.08198533 - samples/sec: 25.05
2020-07-02 20:11:34,515 ----------------------------------------------------------------------------------------------------
2020-07-02 20:11:34,517 EPOCH 87 done: loss 0.0823 - lr 0.0001267
2020-07-02 20:11:57,259 DEV : loss 0.19768422842025757 - score 0.9798
2020-07-02 20:11:57,329 BAD EPOCHS (no improvement): 3
2020-07-02 20:11:57,330 ----------------------------------------------------------------------------------------------------
2020-07-02 20:12:20,421 epoch 88 - iter 15/154 - loss 0.10425307 - samples/sec: 21.03
2020-07-02 20:12:40,927 epoch 88 - iter 30/154 - loss 0.08032743 - samples/sec: 23.56
2020-07-02 20:13:01,191 epoch 88 - iter 45/154 - loss 0.08593136 - samples/sec: 24.04
2020-07-02 20:13:19,940 epoch 88 - iter 60/154 - loss 0.08089487 - samples/sec: 25.77
2020-07-02 20:13:39,682 epoch 88 - iter 75/154 - loss 0.07908025 - samples/sec: 24.45
2020-07-02 20:13:59,018 epoch 88 - iter 90/154 - loss 0.08750883 - samples/sec: 25.15
2020-07-02 20:14:20,234 epoch 88 - iter 105/154 - loss 0.08899166 - samples/sec: 22.75
2020-07-02 20:14:39,067 epoch 88 - iter 120/154 - loss 0.08754672 - samples/sec: 25.65
2020-07-02 20:14:59,203 epoch 88 - iter 135/154 - loss 0.08932739 - samples/sec: 24.17
2020-07-02 20:15:17,676 epoch 88 - iter 150/154 - loss 0.08721906 - samples/sec: 26.15
2020-07-02 20:15:22,756 ----------------------------------------------------------------------------------------------------
2020-07-02 20:15:22,757 EPOCH 88 done: loss 0.0892 - lr 0.0001267
2020-07-02 20:15:45,517 DEV : loss 0.1971229910850525 - score 0.9798
2020-07-02 20:15:45,584 BAD EPOCHS (no improvement): 4
2020-07-02 20:15:45,586 ----------------------------------------------------------------------------------------------------
2020-07-02 20:16:06,439 epoch 89 - iter 15/154 - loss 0.07559101 - samples/sec: 23.33
2020-07-02 20:16:27,847 epoch 89 - iter 30/154 - loss 0.07773159 - samples/sec: 22.54
2020-07-02 20:16:47,195 epoch 89 - iter 45/154 - loss 0.09153157 - samples/sec: 25.20
2020-07-02 20:17:09,002 epoch 89 - iter 60/154 - loss 0.08735939 - samples/sec: 22.14
2020-07-02 20:17:29,072 epoch 89 - iter 75/154 - loss 0.08408470 - samples/sec: 24.06
2020-07-02 20:17:49,062 epoch 89 - iter 90/154 - loss 0.08779174 - samples/sec: 24.17
2020-07-02 20:18:08,900 epoch 89 - iter 105/154 - loss 0.08387496 - samples/sec: 24.35
2020-07-02 20:18:28,500 epoch 89 - iter 120/154 - loss 0.08260459 - samples/sec: 24.82
2020-07-02 20:18:48,390 epoch 89 - iter 135/154 - loss 0.08210002 - samples/sec: 24.34
2020-07-02 20:19:08,537 epoch 89 - iter 150/154 - loss 0.08263975 - samples/sec: 23.97
2020-07-02 20:19:13,248 ----------------------------------------------------------------------------------------------------
2020-07-02 20:19:13,250 EPOCH 89 done: loss 0.0817 - lr 0.0001267
2020-07-02 20:19:35,671 DEV : loss 0.19783227145671844 - score 0.9792
2020-07-02 20:19:35,900 BAD EPOCHS (no improvement): 5
2020-07-02 20:19:35,902 ----------------------------------------------------------------------------------------------------
2020-07-02 20:19:58,748 epoch 90 - iter 15/154 - loss 0.08422335 - samples/sec: 21.28
2020-07-02 20:20:19,410 epoch 90 - iter 30/154 - loss 0.07745877 - samples/sec: 23.37
2020-07-02 20:20:39,526 epoch 90 - iter 45/154 - loss 0.07794650 - samples/sec: 24.20
2020-07-02 20:20:58,368 epoch 90 - iter 60/154 - loss 0.07353494 - samples/sec: 25.64
2020-07-02 20:21:18,494 epoch 90 - iter 75/154 - loss 0.07113823 - samples/sec: 23.98
2020-07-02 20:21:40,280 epoch 90 - iter 90/154 - loss 0.07130840 - samples/sec: 22.29
2020-07-02 20:21:59,776 epoch 90 - iter 105/154 - loss 0.07188892 - samples/sec: 24.76
2020-07-02 20:22:17,909 epoch 90 - iter 120/154 - loss 0.07002038 - samples/sec: 26.84
2020-07-02 20:22:38,265 epoch 90 - iter 135/154 - loss 0.07641873 - samples/sec: 23.71
2020-07-02 20:22:58,198 epoch 90 - iter 150/154 - loss 0.07757820 - samples/sec: 24.24
2020-07-02 20:23:02,778 ----------------------------------------------------------------------------------------------------
2020-07-02 20:23:02,779 EPOCH 90 done: loss 0.0785 - lr 0.0001267
2020-07-02 20:23:25,603 DEV : loss 0.19686871767044067 - score 0.9798
Epoch    90: reducing learning rate of group 0 to 6.3352e-05.
2020-07-02 20:23:25,672 BAD EPOCHS (no improvement): 6
2020-07-02 20:23:25,674 ----------------------------------------------------------------------------------------------------
2020-07-02 20:23:25,675 ----------------------------------------------------------------------------------------------------
2020-07-02 20:23:25,675 learning rate too small - quitting training!
2020-07-02 20:23:25,676 ----------------------------------------------------------------------------------------------------
2020-07-02 20:23:26,199 ----------------------------------------------------------------------------------------------------
2020-07-02 20:23:26,201 Testing using best model ...
2020-07-02 20:23:26,204 loading file Path/to/model/output/directory/best-model.pt
2020-07-02 20:23:40,895 0.974	0.974	0.974
2020-07-02 20:23:40,897 
MICRO_AVG: acc 0.9913333333333333 - f1-score 0.974
MACRO_AVG: acc 0.9913333333333334 - f1-score 0.9776831958334483
ABBR       tp: 9 - fp: 0 - fn: 0 - tn: 491 - precision: 1.0000 - recall: 1.0000 - accuracy: 1.0000 - f1-score: 1.0000
DESC       tp: 136 - fp: 5 - fn: 2 - tn: 357 - precision: 0.9645 - recall: 0.9855 - accuracy: 0.9860 - f1-score: 0.9749
ENTY       tp: 86 - fp: 3 - fn: 8 - tn: 403 - precision: 0.9663 - recall: 0.9149 - accuracy: 0.9780 - f1-score: 0.9399
HUM        tp: 63 - fp: 1 - fn: 2 - tn: 434 - precision: 0.9844 - recall: 0.9692 - accuracy: 0.9940 - f1-score: 0.9767
LOC        tp: 80 - fp: 1 - fn: 1 - tn: 418 - precision: 0.9877 - recall: 0.9877 - accuracy: 0.9960 - f1-score: 0.9877
NUM        tp: 113 - fp: 3 - fn: 0 - tn: 384 - precision: 0.9741 - recall: 1.0000 - accuracy: 0.9940 - f1-score: 0.9869
2020-07-02 20:23:40,898 ----------------------------------------------------------------------------------------------------
```
</details>

The model was saved in the directory `OUTPUT_DIR`.  We can load the sequence classifier into our `EasySequenceClassifier`
instance and start running inference.

```python
from adaptnlp import EasySequenceClassifier
# Set example text and instantiate tagger instance
example_text = '''Where was the Queen's wedding held? '''

classifier = EasySequenceClassifier()

sentences = classifier.tag_text(example_text, model_name_or_path=OUTPUT_DIR)
print("Label output:\n")
for sentence in sentences:
    print(sentence.labels)

```
<details class = "summary">
<summary>Output</summary>
```python
[LOC (0.9990556836128235)]
```
</details>

