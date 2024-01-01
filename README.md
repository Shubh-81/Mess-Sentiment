# Few-shot Sentiment Classification with Ollama, Llama2, and LangChain

## Author
Shubh Agarwal (21110205)

## Published Date
December 30, 2023

## Overview
This repository demonstrates the implementation of few-shot sentiment classification using Ollama, Llama2, and LangChain. The code utilizes the LangChain package to streamline the process and includes examples for training and evaluating the model.

## Setup
Ensure you have the required dependencies installed:
```bash
pip install langchain
```

## Code Structure
The primary code is in the notebook (`main.ipynb`), which includes the following sections:

### 1. Importing Modules
The notebook begins by importing necessary modules from the LangChain package, including Ollama for running large language models locally, CallbackManager for managing callbacks during training or inference, StreamingStdOutCallbackHandler for real-time output streaming, and various classes for defining templates and chains.

### 2. Initializing Ollama LLM Instance
An Ollama instance is created with the LLaMA-2 model, configured to use StreamingStdOutCallbackHandler for streaming predictions.

### 3. Defining the Labeled Dataset
A labeled dataset for training is defined using examples, each containing a phrase and its corresponding class (Alpha or Beta).

### 4. Template Definition
A template (`example_template`) is created using Jinja2 syntax to format labeled examples for few-shot learning.

### 5. Creating a FewShotPromptTemplate
A FewShotPromptTemplate is defined using the template, examples, and other parameters to generate prompts for few-shot learning.

### 6. Prompt Generation and Model Evaluation
Prompt generation is demonstrated with a sample input phrase. The LLMChain is created using the Ollama instance and the FewShotPromptTemplate. Predictions are then made for different input sentences.

### 7. Evaluation Summary
The notebook concludes with an evaluation summary of the Llama2 model based on the provided examples. Observations and challenges are highlighted, emphasizing the model's robustness in classifying sentiments but also noting limitations in handling mixed sentiments and sarcasm.
