This advanced tutorial section goes over using AdaptNLP for training and fine-tuning your own custom NLP models
to get State-of-the-Art results.

You should ideally follow the tutorials along with the provided notebooks in the `tutorials` directory at the top
level of the AdaptNLP library.

You could also run the code snippets in these tutorials straight through the python interpreter as well.

## Install and Setup

AdaptNLP can be used with or without GPUs.  AdaptNLP will automatically make use of GPU VRAM in environment with
CUDA-compatible NVIDIA GPUs and NVIDIA drivers installed.  GPU-less environments will run AdaptNLP modules fine as well.

You will almost always want to utilize GPUs for training and fine-tuning useful NLP models, so a CUDA-compatible NVIDIA
GPU is a must.

Multi-GPU environments with Apex installed can allow for distributed and/or mixed precision training.

## Overview of Training and Finetuning Capabilities

Simply training a state-of-the-art sequence classification model can be done with AdaptNLP using Flair's sequence 
classification model and trainer with general pre-trained language models.  With encoders providing accurate word 
representations via. models like ALBERT, GPT2, and other transformer models, we can produce accurate NLP task-related
models.

With the concepts of [ULMFiT](https://arxiv.org/abs/1801.06146) in mind, AdaptNLP's approach in training downstream
predictive NLP models like sequence classification takes a step further than just utilizing pre-trained
contextualized embeddings.  We are able to effectively fine-tune state-of-the-art language models for useful NLP
tasks on various domain specific data.

##### Training a Sequence Classification with `SequenceClassifierTrainer`

```python
from adaptnlp import EasyDocumentEmbeddings, SequenceClassifierTrainer
from flair.datasets import TREC_6

# Specify directory for trainer files and model to be downloaded to
OUTPUT_DIR = "Path/to/model/output/directory" 

# Load corpus and instantiate AdaptNLP's `EasyDocumentEmbeddings` with desired embeddings
corpus = TREC_6() # Or path to directory of train.csv, test.csv, dev.csv files at "Path/to/data/directory" 
doc_embeddings = EasyDocumentEmbeddings("bert-base-cased", methods=["rnn"])

# Instantiate the trainer for Sequence Classification with the dataset, embeddings, and mapping of column index of data
sc_trainer = SequenceClassifierTrainer(corpus=corpus, encoder=doc_embeddings)

# Find optimal learning rate with automated learning rate finder
learning_rate = sc_trainer.find_learning_rate(output_dir=OUTPUT_DIR)

# Train the sequence classifier
sc_trainer.train(output_dir=OUTPUT_DIR, learning_rate=learning_rate, mini_batch_size=32, max_epochs=150)

# Now load the `EasySequenceClassifier` with the path to your trained model and run inference on your text.
from adaptnlp import EasySequenceClassifier

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
2020-07-02 13:56:09,357 [b'LOC', b'DESC', b'ENTY', b'HUM', b'NUM', b'ABBR']

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

Recommended Learning Rate 0.019952623149688778
2020-07-02 13:58:10,067 ----------------------------------------------------------------------------------------------------
2020-07-02 13:58:10,071 Model: "TextClassifier(
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
2020-07-02 13:58:10,071 ----------------------------------------------------------------------------------------------------
2020-07-02 13:58:10,073 Corpus: "Corpus: 4907 train + 545 dev + 500 test sentences"
2020-07-02 13:58:10,074 ----------------------------------------------------------------------------------------------------
2020-07-02 13:58:10,074 Parameters:
2020-07-02 13:58:10,075  - learning_rate: "0.019952623149688778"
2020-07-02 13:58:10,076  - mini_batch_size: "32"
2020-07-02 13:58:10,077  - patience: "5"
2020-07-02 13:58:10,077  - anneal_factor: "0.5"
2020-07-02 13:58:10,080  - max_epochs: "150"
2020-07-02 13:58:10,081  - shuffle: "True"
2020-07-02 13:58:10,081  - train_with_dev: "False"
2020-07-02 13:58:10,082  - batch_growth_annealing: "False"
2020-07-02 13:58:10,083 ----------------------------------------------------------------------------------------------------
2020-07-02 13:58:10,084 Model training base path: "data-volume-1"
2020-07-02 13:58:10,084 ----------------------------------------------------------------------------------------------------
2020-07-02 13:58:10,087 Device: cpu
2020-07-02 13:58:10,088 ----------------------------------------------------------------------------------------------------
2020-07-02 13:58:10,088 Embeddings storage mode: cpu
2020-07-02 13:58:10,091 ----------------------------------------------------------------------------------------------------
2020-07-02 13:58:32,513 epoch 1 - iter 15/154 - loss 2.13271559 - samples/sec: 21.61
2020-07-02 13:58:53,034 epoch 1 - iter 30/154 - loss 2.05371269 - samples/sec: 23.78
2020-07-02 13:59:12,643 epoch 1 - iter 45/154 - loss 2.01068912 - samples/sec: 24.63
2020-07-02 13:59:33,185 epoch 1 - iter 60/154 - loss 2.00240319 - samples/sec: 23.50
2020-07-02 13:59:52,275 epoch 1 - iter 75/154 - loss 1.97219250 - samples/sec: 25.29
2020-07-02 14:00:11,458 epoch 1 - iter 90/154 - loss 1.95007052 - samples/sec: 25.18
2020-07-02 14:00:31,200 epoch 1 - iter 105/154 - loss 1.91979002 - samples/sec: 24.46
2020-07-02 14:00:51,149 epoch 1 - iter 120/154 - loss 1.90456727 - samples/sec: 24.38
2020-07-02 14:01:11,720 epoch 1 - iter 135/154 - loss 1.87982891 - samples/sec: 23.49
2020-07-02 14:01:31,615 epoch 1 - iter 150/154 - loss 1.85678710 - samples/sec: 24.27
2020-07-02 14:01:36,154 ----------------------------------------------------------------------------------------------------
2020-07-02 14:01:36,156 EPOCH 1 done: loss 1.8501 - lr 0.0199526
2020-07-02 14:01:57,790 DEV : loss 1.5790961980819702 - score 0.7859
2020-07-02 14:01:58,060 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 14:01:58,574 ----------------------------------------------------------------------------------------------------
2020-07-02 14:02:20,435 epoch 2 - iter 15/154 - loss 1.63866882 - samples/sec: 22.17
2020-07-02 14:02:40,418 epoch 2 - iter 30/154 - loss 1.61095683 - samples/sec: 24.17
2020-07-02 14:02:59,702 epoch 2 - iter 45/154 - loss 1.58390728 - samples/sec: 25.28
2020-07-02 14:03:20,338 epoch 2 - iter 60/154 - loss 1.57619473 - samples/sec: 23.40
2020-07-02 14:03:39,984 epoch 2 - iter 75/154 - loss 1.54124361 - samples/sec: 24.58
2020-07-02 14:03:59,982 epoch 2 - iter 90/154 - loss 1.51295740 - samples/sec: 24.33
2020-07-02 14:04:19,493 epoch 2 - iter 105/154 - loss 1.48691362 - samples/sec: 24.75
2020-07-02 14:04:38,575 epoch 2 - iter 120/154 - loss 1.47600422 - samples/sec: 25.34
2020-07-02 14:04:58,640 epoch 2 - iter 135/154 - loss 1.45883828 - samples/sec: 24.23
2020-07-02 14:05:19,545 epoch 2 - iter 150/154 - loss 1.44938952 - samples/sec: 23.09
2020-07-02 14:05:23,818 ----------------------------------------------------------------------------------------------------
2020-07-02 14:05:23,820 EPOCH 2 done: loss 1.4470 - lr 0.0199526
2020-07-02 14:05:45,338 DEV : loss 1.3960844278335571 - score 0.7994
2020-07-02 14:05:45,405 BAD EPOCHS (no improvement): 0
saving best model
2020-07-02 14:05:49,806 ----------------------------------------------------------------------------------------------------
2020-07-02 14:06:11,158 epoch 3 - iter 15/154 - loss 1.19971250 - samples/sec: 22.72
2020-07-02 14:06:31,952 epoch 3 - iter 30/154 - loss 1.23841848 - samples/sec: 23.22
2020-07-02 14:06:52,692 epoch 3 - iter 45/154 - loss 1.22383659 - samples/sec: 23.46
2020-07-02 14:07:12,481 epoch 3 - iter 60/154 - loss 1.22022739 - samples/sec: 24.40
2020-07-02 14:07:33,006 epoch 3 - iter 75/154 - loss 1.21155183 - samples/sec: 23.51
2020-07-02 14:07:51,805 epoch 3 - iter 90/154 - loss 1.19975713 - samples/sec: 25.92
2020-07-02 14:08:10,120 epoch 3 - iter 105/154 - loss 1.18502910 - samples/sec: 26.38
2020-07-02 14:08:31,909 epoch 3 - iter 120/154 - loss 1.17209500 - samples/sec: 22.15
2020-07-02 14:08:52,007 epoch 3 - iter 135/154 - loss 1.14961906 - samples/sec: 24.02
2020-07-02 14:09:11,329 epoch 3 - iter 150/154 - loss 1.14153116 - samples/sec: 24.99
2020-07-02 14:09:15,868 ----------------------------------------------------------------------------------------------------
2020-07-02 14:09:15,869 EPOCH 3 done: loss 1.1420 - lr 0.0199526
2020-07-02 14:09:37,671 DEV : loss 1.3099236488342285 - score 0.7939
2020-07-02 14:09:37,737 BAD EPOCHS (no improvement): 1
2020-07-02 14:09:37,739 ----------------------------------------------------------------------------------------------------
2020-07-02 14:10:00,789 epoch 4 - iter 15/154 - loss 0.97922458 - samples/sec: 21.04
2020-07-02 14:10:21,158 epoch 4 - iter 30/154 - loss 0.99391881 - samples/sec: 23.94
2020-07-02 14:10:41,210 epoch 4 - iter 45/154 - loss 0.96832921 - samples/sec: 24.09
2020-07-02 14:11:00,520 epoch 4 - iter 60/154 - loss 0.96508796 - samples/sec: 25.00
2020-07-02 14:11:21,664 epoch 4 - iter 75/154 - loss 0.97899694 - samples/sec: 22.98
```
</details>

##### Fine-Tuning a Transformers Language Model with `LMFineTuner`


```python
from adaptnlp import LMFineTuner

# Set output directory to store fine-tuner files and models
OUTPUT_DIR = "Path/to/model/output/directory" 

# Set path to train.csv and test.csv datasets, must have a column header labeled "text" to specify data to train language model
train_data_file = "Path/to/train.csv" 
eval_data_file = "Path/to/test.csv"

finetuner = LMFineTuner(
                        train_data_file=train_data_file,
                        eval_data_file=eval_data_file,
                        model_type="bert",
                        model_name_or_path="bert-base-cased",
                        mlm=True
                        )
# Freeze layers up to the last group of classification layers
finetuner.freeze()

# Find optimal learning rate with automated learning rate finder
learning_rate = finetuner.find_learning_rate(base_path=OUTPUT_DIR)
finetuner.freeze()

finetuner.train_one_cycle(
                          output_dir=OUTPUT_DIR,
                          learning_rate=learning_rate,
                          per_gpu_train_batch_size=4,
                          num_train_epochs=10.0,
                          evaluate_during_training=True,
                          )
```