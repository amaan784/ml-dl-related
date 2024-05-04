# Hugging Face

- Note- 
- A lot of storage space may be needed since the models are heavy 

## Pipeline Function-

The pipeline function returns an end to end object that performs an NLP task on one or more several texts.

Works in 3 steps -
1) pre-processing the text(s) -  applies tokenizer to the model
2) feeds the pre processed text to the model 
3) post processing - shows results

Workflow-

--> pre-process --> Model --> post process -->

The Pipeline API supports most common NLP tasks-
- Text Classification
- Zero Shot Classification
- Text Generation
- Text Completion (mask filling)
- Token Classification
- Question Answering
- Summarization
- Translation

## Tokenizer

- Transforms raw text to numbers
- Objective to find the most meaningful representation

## Pytorch / Tensorflow

- Both can be used with hugging face

## Save / Load

- Model and the tokenzier can be saved and loaded

## Model Hub

- Contains a large number of models serving different purposes

## Fine tune

- Approach to transfer learning in which the parameters of a pre-trained model are trained on new data

General Steps-
1) Prepare datset
2) Load pretrained tokenizer, call it with dataset and get encoding
3) Build tensorflow / pytorch dataset with encodings
4) Load pre trained model
5) a) Load Trainer class and train int
    b) native pytorch training loop / tensorflow code

## Sources-
- [Getting Started With Hugging Face in 15 Minutes | Transformers, Pipeline, Tokenizer, Models](https://youtu.be/QEaBAZQCtwE)
- https://huggingface.co/docs
- https://huggingface.co/learn/nlp-course/chapter1/1