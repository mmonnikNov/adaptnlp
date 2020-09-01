# Text Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Novetta/adaptnlp/blob/master/tutorials/7.%20Text%20Generation/Easy%20Text%20Generator.ipynb)


Text generation is the NLP task of generating a coherent sequence of words, usually from a language model. The current leading methods, most notably OpenAI’s GPT-2 and GPT-3, rely on feeding tokens (words or characters) into a pre-trained language model which then uses this seed data to construct a sequence of text. AdaptNLP provides simple methods to easily fine-tune these state-of-the-art models and generate text for any use case. 

Below, we'll walk through how we can use AdaptNLP's `EasyTextGenerator` module to generate text to complete a given String.


## Getting Started with `TextGeneration`

We'll first get started by importing the `EasyTextGenerator` class from AdaptNLP.

After that, we set some example text that we'll use further down, and then instantiate the generator.

```python
from adaptnlp import EasyTextGenerator

# Text from encyclopedia Britannica on Einstein
text = "What has happened?"

generator = EasyTextGenerator()
```

### Generate with `generate(text:str, model_name_or_path: str, mini_batch_size: int, num_tokens_to_produce: int **kwargs)`
Now that we have the summarizer instantiated, we are ready to load in a model and compress the text 
with the built-in `generate()` method.  

This method takes in parameters: `text`, `model_name_or_path`, `mini_batch_size`, and `num_tokens_to_produce` as well as optional keyword arguments
from the `Transformers.PreTrainedModel.generate()` method.

!!! note 
    You can set `model_name_or_path` to any of Transformers pretrained Generation Models with Language Model heads.
    Transformers models are located at [https://huggingface.co/models](https://huggingface.co/models).  You can also pass in
    the path of a custom trained Transformers `xxxWithLMHead` model.
 
The method returns a list of Strings.

Here is one example using the gpt2 model:

```python
# Generate
generated_text = generator.generate("What has happened", model_name_or_path="gpt2", mini_batch_size=2, num_tokens_to_produce=50)

print(generated_text)
```
<details class = "summary">
<summary>Output</summary>
```
["What has happened to the world's most important technology?\n\nThe world's most important technology is the Internet. It's the most important technology in the world. It's the most important technology in the world. It's the most important technology in the world."]
```
</details>
