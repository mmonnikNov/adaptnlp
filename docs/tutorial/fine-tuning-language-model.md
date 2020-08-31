# What is a language model?

Language modeling is the task of generating a probability distribution over a sequence of words. The language models that we are using can assign the probabilitiy of an upcoming word(s) given a sequence of words. The GPT2 language model is a good example of a *Causal Language Model* which can predict words following a sequence of words. This predicted word can then be used along the given sequence of words to predict another word and so on. This is how we actually a variant of how we produce models for the NLP task of text generation.

# Why would you want to fine-tune a language model? 

Fine-tuning a language model comes in handy when data of a target task comes from a different distribution compared to the general-domain data that was used for pretraining a language model.

When fine-tuning the language model on data from a target task, the general-domain pretrained model is able to converge
quickly and adapt to the idiosyncrasies of the target data.  This can be seen from the efforts of ULMFiT and Jeremy
Howard's and Sebastian Ruder's approach on NLP transfer learning.

With AdaptNLP's `LMFineTuner`, we can start to fine-tune state-of-the-art pretrained transformers architecture 
language models provided by Hugging Face's Transformers library. `LMFineTuner` is built on `transformers.Trainer` so additional documentation on it can be found at Hugging Face's documentation [here](https://huggingface.co/transformers/master/main_classes/trainer.html)

Below are the available transformers language models for fine-tuning with `LMFineTuner`

| Transformer Model| Model Type/Architecture String Key|
| ------------- | ----------------------  |
| ALBERT | "albert" |
| DistilBERT | "distilbert" |
| BERT | "bert" |
| CamemBERT | "camembert" |
| RoBERTa | "roberta" |
| GPT | "gpt" |
| GPT2 | "gpt2" |

You can fine-tune on any transformers language models with the above architecture in Huggingface's Transformers
library.  Key shortcut names are located [here](https://huggingface.co/transformers/pretrained_models.html).

The same goes for Huggingface's public model-sharing repository, which is available [here](https://huggingface.co/models)
as of v2.2.2 of the Transformers library.

This tutorial will go over the following simple-to-use componenets of using the `LMFineTuner` to fine-tune pre-trained language models on your custom text data.
1. Data loading and training arguments
2. Language model training
3. Language model evaluation

# 1. Data loading and training arguments

We'll first start by downloading some example raw text files. If you want to fine-tune a model on your own custom data, just provide the file paths to the training and evaluation text files that contain text from your target task. You don't require a lot of formatting with the data since a language model does not necessarily require "labeled" data. All you need is the text you'd like use to "expand" the domain of knowledge that your language model is training on.


```python
!wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
!unzip wikitext-2-raw-v1.zip

train_file = "./wikitext-2-raw/wiki.train.raw"
eval_file = "./wikitext-2-raw/wiki.test.raw"
```

<details class = "summary">
<summary>Output</summary>
```python
    --2020-08-31 15:38:50--  https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.217.64.78
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.217.64.78|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4721645 (4.5M) [application/zip]
    Saving to: ‘wikitext-2-raw-v1.zip’
    
    wikitext-2-raw-v1.z 100%[===================>]   4.50M  2.92MB/s    in 1.5s    
    
    2020-08-31 15:38:52 (2.92 MB/s) - ‘wikitext-2-raw-v1.zip’ saved [4721645/4721645]
    
    Archive:  wikitext-2-raw-v1.zip
       creating: wikitext-2-raw/
      inflating: wikitext-2-raw/wiki.test.raw  
      inflating: wikitext-2-raw/wiki.valid.raw  
      inflating: wikitext-2-raw/wiki.train.raw  
```
</details>


Now that we have the text data we want to fine-tune our language model on, we can move on to configuring the training component.

One of the first things we'll need to specify before we start training are the training arguments. Training arguments consist mainly of the hyperparameters we want to provide the model. These may include batch size, initial learning rate, number of epochs, etc.

We will be using the `transformers.TrainingArguments` data class to store our training args. These are compatible with the `transformers.Trainer` as well as AdaptNLP's train methods. For more documention on the `TrainingArguments` class, please look [here](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments). There are a lot of arguments available, but we will pass in the important args and use default values for the rest.

The training arguments below specify the output directory for you model and checkpoints. 


```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./models',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=False,
    logging_dir='./logs',
    save_steps=2500,
    eval_steps=100
)
```

# 2. Language model training

Now that we have our data and training arguments, let's instantiate the `LMFineTuner` and load in a pre-trained language model we would like to fine-tune. In this case, we will use the `gpt2` pre-trained language model.

Note: You can load in any model with the allowable architecture that we've specified above. You can even load in custom pre-trained models or models that you find in the Hugging Face repository that have already been fine-tuned and trained on NLP target tasks.


```python
from adaptnlp import LMFineTuner

finetuner = LMFineTuner(model_name_or_path="gpt2")

```

Now we can run the built-in `train()` method by passing in the training arguments. The training method will also be where you specify your data arguments which include the your train and eval datasets, the pre-trained model ID (this should have been loaded from your earlier cells, but can be loaded dynamically), text column name, label column name, and ordered label names (only required if loading in paths to CSV data file for dataset args).

Notice how we pass the `mlm` argument as False? The mlm argument should be true if we are using a masked language model variant such as BERT architecture language models. More information can be found on Hugging Face's documentation [here](https://huggingface.co/transformers/master/task_summary.html#masked-language-modeling)

Please checkout AdaptNLP's package reference for more information [here](https://novetta.github.io/adaptnlp/class-api/language-model-module.html).


```python
finetuner.train(
    training_args=training_args,
    train_file=eval_file,
    eval_file=eval_file,
    mlm=False,
    overwrite_cache=False
)
```
<details class = "summary">
<summary>Output</summary>
```python
    08/31/2020 15:45:44 - INFO - transformers.training_args -   PyTorch: setting up devices
    08/31/2020 15:45:44 - WARNING - adaptnlp.language_model -   Process rank: -1,
                    device: cuda:0,
                    n_gpu: 1,
                    distributed training: False,
                    16-bits training: False
                
    08/31/2020 15:45:44 - INFO - adaptnlp.language_model -   Training/evaluation parameters: {
      "output_dir": "./models",
      "overwrite_output_dir": false,
      "do_train": false,
      "do_eval": false,
      "do_predict": false,
      "evaluate_during_training": false,
      "per_device_train_batch_size": 1,
      "per_device_eval_batch_size": 1,
      "per_gpu_train_batch_size": null,
      "per_gpu_eval_batch_size": null,
      "gradient_accumulation_steps": 1,
      "learning_rate": 5e-05,
      "weight_decay": 0.01,
      "adam_epsilon": 1e-08,
      "max_grad_norm": 1.0,
      "num_train_epochs": 1,
      "max_steps": -1,
      "warmup_steps": 500,
      "logging_dir": "./logs",
      "logging_first_step": false,
      "logging_steps": 500,
      "save_steps": 2500,
      "save_total_limit": null,
      "no_cuda": false,
      "seed": 42,
      "fp16": false,
      "fp16_opt_level": "O1",
      "local_rank": -1,
      "tpu_num_cores": null,
      "tpu_metrics_debug": false,
      "debug": false,
      "dataloader_drop_last": false,
      "eval_steps": 100,
      "past_index": -1
    }
    08/31/2020 15:45:44 - INFO - filelock -   Lock 139826145788648 acquired on ./wikitext-2-raw/cached_lm_GPT2TokenizerFast_1024_wiki.test.raw.lock
    08/31/2020 15:45:44 - INFO - transformers.data.datasets.language_modeling -   Creating features from dataset file at ./wikitext-2-raw
    08/31/2020 15:45:45 - INFO - transformers.data.datasets.language_modeling -   Saving features into cached file ./wikitext-2-raw/cached_lm_GPT2TokenizerFast_1024_wiki.test.raw [took 0.004 s]
    08/31/2020 15:45:45 - INFO - filelock -   Lock 139826145788648 released on ./wikitext-2-raw/cached_lm_GPT2TokenizerFast_1024_wiki.test.raw.lock
    08/31/2020 15:45:45 - INFO - filelock -   Lock 139826145788312 acquired on ./wikitext-2-raw/cached_lm_GPT2TokenizerFast_1024_wiki.test.raw.lock
    08/31/2020 15:45:45 - INFO - transformers.data.datasets.language_modeling -   Loading features from cached file ./wikitext-2-raw/cached_lm_GPT2TokenizerFast_1024_wiki.test.raw [took 0.006 s]
    08/31/2020 15:45:45 - INFO - filelock -   Lock 139826145788312 released on ./wikitext-2-raw/cached_lm_GPT2TokenizerFast_1024_wiki.test.raw.lock
    08/31/2020 15:45:45 - INFO - transformers.trainer -   You are instantiating a Trainer but W&B is not installed. To use wandb logging, run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface.
    08/31/2020 15:45:45 - INFO - transformers.trainer -   ***** Running training *****
    08/31/2020 15:45:45 - INFO - transformers.trainer -     Num examples = 279
    08/31/2020 15:45:45 - INFO - transformers.trainer -     Num Epochs = 1
    08/31/2020 15:45:45 - INFO - transformers.trainer -     Instantaneous batch size per device = 1
    08/31/2020 15:45:45 - INFO - transformers.trainer -     Total train batch size (w. parallel, distributed & accumulation) = 1
    08/31/2020 15:45:45 - INFO - transformers.trainer -     Gradient Accumulation steps = 1
    08/31/2020 15:45:45 - INFO - transformers.trainer -     Total optimization steps = 279
    Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
    Iteration: 100%|██████████| 279/279 [01:04<00:00,  4.33it/s][A
    Epoch: 100%|██████████| 1/1 [01:04<00:00, 64.48s/it]
    08/31/2020 15:46:49 - INFO - transformers.trainer -   
    
    Training completed. Do not forget to share your model on huggingface.co/models =)
    
    
    08/31/2020 15:46:49 - INFO - transformers.trainer -   Saving model checkpoint to ./models
    08/31/2020 15:46:49 - INFO - transformers.configuration_utils -   Configuration saved in ./models/config.json
    08/31/2020 15:46:50 - INFO - transformers.modeling_utils -   Model weights saved in ./models/pytorch_model.bin
```
</details>


#  3. Language model evaluation

To run evaluation on the model with your eval dataset, all you need to call is the built-in `finetuner.evaluate()`, since you've already loaded in your eval dataset during training.


```python
finetuner.evaluate()
```

And now you have your very own pre-trained language model that's been fine-tuned on your personal domain data!

Since we've just fine-tuned a causal language model, we can actually load this straight into an `EasyTextGenerator` class object and play around with our language model to evaluate it qualitatively with our own "eyes".

All we have to do is pass in the directory that we've output our trained language model, in this case it's located in "./models"


```python
from adaptnlp import EasyTextGenerator

text = "China and the U.S. will begin to"

generator = EasyTextGenerator()
```


```python
# Generate
generated_text = generator.generate(
    text, 
    model_name_or_path="./models", 
    num_tokens_to_produce=50
)

print(generated_text)

```
<details class = "summary">
<summary>Output</summary>
```python

    Generating: 100%|██████████| 1/1 [00:00<00:00,  2.26it/s]

    ['China and the U.S. will begin to develop their own nuclear weapons in the coming years.\n\nThe U.S. has been developing a range of nuclear weapons since the 1950s, but the U.S. has never used them in combat. The U.S. has been']
```
</details>


    


You can compare this with the original pre-trained gpt2 model as well. 


```python
# Generate
generated_text = generator.generate(
    text, 
    model_name_or_path="gpt2", 
    num_tokens_to_produce=50
)

print(generated_text)

```

<details class = "summary">
<summary>Output</summary>
```python
    Special tokens have been added in the vocabulary, make sure the associated word emebedding are fine-tuned or trained.
    Some weights of GPT2LMHeadModel were not initialized from the model checkpoint at gpt2 and are newly initialized: ['h.0.attn.masked_bias', 'h.1.attn.masked_bias', 'h.2.attn.masked_bias', 'h.3.attn.masked_bias', 'h.4.attn.masked_bias', 'h.5.attn.masked_bias', 'h.6.attn.masked_bias', 'h.7.attn.masked_bias', 'h.8.attn.masked_bias', 'h.9.attn.masked_bias', 'h.10.attn.masked_bias', 'h.11.attn.masked_bias', 'lm_head.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Generating: 100%|██████████| 1/1 [00:00<00:00,  2.33it/s]

    ['China and the U.S. will begin to see the effects of the new sanctions on the Russian economy.\n\n"The U.S. is going to be the first to see the effects of the new sanctions," said Michael O\'Hanlon, a senior fellow at the Center for Strategic']
```

    

