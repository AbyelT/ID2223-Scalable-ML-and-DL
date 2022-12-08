# Lab 2: Fine-Tune a Transformer model for Language Transcription

## Intro
The purpose with this lab is to fine-tune a pre-trained model for automatic speech recognition (ASR), for any multilingual data. This lab features the pre-trained model Whisper (a checkpoint of the model) and uses the Common Voice dataset for fine-tuning the model, to achieve stronger performance. The lab has the following tasks:

1. Fine-tune the Whispser transformer model for the Swedish
language
2. Build and run an inference pipeline with a Gradio UI on Hugging Face Spaces, for demonstrating the model.
3. Discuss possible improvement to the model performance e.g. model-centric or data-centric approach and apply it to a new model
4. Refactor the program into a feature engineering pipeline, training pipeline, and an inference program

## Tools
- Pre-trained Transformer model: Whisper
- Dataset: [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
- Data preparation, training and fine-tuning: Google colab
- Inference UI: Hugging Face Spaces 
- Model storage: Hugging Face model repository

## Steps

The steps taken in the lab follows the guide on how to fine-tune the Whisper model for any multilingual ASR dataset, by [Sandchit Gandhi](https://huggingface.co/blog/fine-tune-whisper). The steps have been refactored for better scalability and readability, the notebook consist of the following steps: 

- Prepare environment: This means downloading the necessary dependencies and modules for the later steps. The user must also authenticate to Hugging Face, in order to upload any checkpoints and the trained model later.

- Feature extraction pipeline: This step uses the CPU and is split into two different steps. One (2a) for extracting the features from the dataset first time, preparing the data takes time so this step also includes saving data to storage (either hugging Face or Drive) for future training. The other step (2b) only prepares the feature extractor, tokenizer and processor and can be done if the user has executed step 2a at least once.

- Training pipeline: This step is computational heavy and requires the user to enable GPU for training the model, as well as fetching the prepared dataset from storage. Aside from the model the data collator and evaluation metric is also defined here.

- Model inference program: This final step builds a demo in which speech can be recorded and given as input the fine-tuned model to transcribe the corresponding text

# Model 1: Base-case fine-tuned model

In the first iteration, we fine-tuned the small checkpoint of the Whisper pre-trained model with the full Common Voice dataset in the swedish language. This iteration was computation intensive, requiring two sessions from colab to complete the model training. The iteration was one  The following evaluation was achieved:

eval/loss: 0.30
eval/wer: 19.89

There were some factors that may have affected the model score. One is that when resuming the training of the model, the checkpoints from the Whisper-small was used and not the ones that came with the checkpoint. When testing the inference of the model, it did a acceptable in transcribing some sample audio. (see log_base.csv)

## Improving pipeline scalability and model performance

The initial model showed acceptable results with the base case checkpoint and common voice dataset, and could improve by e.g. fine-tuning hyper-parameters or train models with new data sources. The following two ways are currently used to train and improve models:

- Model-centric approach: Collect any available data and develop a model that is good enough to handle any noise. The data is fixed and the focus lies on iteratively improving the model. 
- Data-centric approach: Focus lines on the quality of the data, thus you try to improve the quality through data preparation methods such thatthe data is reliable and efficient. With good data multiple models should be able to perform well.

For improving the model, we chose the *Data-centric approach*. The reason being that training a model is only a small part of the ML lifecyle, while data collection and preparation is a much larger domain. Improving the quality of the data would not only increase the  accuracy of the trained model but also multiple other models, in contrast to optimizing a single model to perform well on poorly prepared data with noise.

# Model 2: Improved model (WIP)

## Speech transcription model
Link: https://huggingface.co/AbyelT/Whisper-models
