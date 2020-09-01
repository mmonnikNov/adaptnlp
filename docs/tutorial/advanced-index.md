This advanced tutorial section goes over using AdaptNLP for training and fine-tuning your own custom NLP models
to get State-of-the-Art results.

You should ideally follow the tutorials along with the provided notebooks in the `tutorials` directory at the top
level of the AdaptNLP library.

You could also run the code snippets in these tutorials straight through the python interpreter as well.

## Install and Setup

AdaptNLP can be used with or without GPUs. AdaptNLP will automatically make use of GPU VRAM in environment with
CUDA-compatible NVIDIA GPUs and NVIDIA drivers installed. GPU-less environments will run AdaptNLP modules fine as well.

You will almost always want to utilize GPUs for training and fine-tuning useful NLP models, so a CUDA-compatible NVIDIA
GPU is a must.

Multi-GPU environments with Apex installed can allow for distributed and/or mixed precision training.

## Overview of Training and Finetuning Capabilities

Downstream NLP-task realted models can be trained with encoders providing accurate word representations via. pre-trained language models like ALBERT, GPT2, and other transformer models.

With the concepts of [ULMFiT](https://arxiv.org/abs/1801.06146) in mind, AdaptNLP's approach in training downstream
predictive NLP models like sequence classification takes a step further than just utilizing pre-trained
contextualized embeddings. We are able to effectively fine-tune state-of-the-art language models for useful NLP tasks on various domain specific data with the help of the `adaptnlp.LMFineTuner` class and the trainers that are provided in our "Easy" modules like `adaptnlp.EasySequenceClassifier`.
