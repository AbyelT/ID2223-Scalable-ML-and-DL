# Lab 2: Fine-Tune a Transformer model for Language Transcription

## Intro
The purpose with this lab is to fine-tune a pre-trained model for automatic speech recognition (ASR), for any multilingual data. This lab features the pre-trained model Whisper (a checkpoint of the model) and uses the Common Voice dataset for fine-tuning the model, to achieve stronger performance. The lab has the following tasks:

1. Fine-tune the Whisper transformer model for the Swedish
language
2. Build and run an inference pipeline with a Gradio UI on Hugging Face Spaces, for demonstrating the model.
3. Discuss possible improvement to the model performance e.g. model-centric or data-centric approach and apply it to a new model.
4. Refactor the program into a feature engineering pipeline, training pipeline, and an model inference program.

## Tools
- Pre-trained Transformer model: Whisper
- Dataset: [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
- Data preparation, training and fine-tuning: Google colab
- Inference UI: Hugging Face Spaces 
- Model storage: Hugging Face model repository

## Steps

The steps taken in the lab follows the guide on how to fine-tune the Whisper model for any multilingual ASR dataset, by [Sandchit Gandhi](https://huggingface.co/blog/fine-tune-whisper). The first model was created as a result of these steps. Later in task 2, the steps was refactored for better scalability and readability

# Model 1: Base-case fine-tuned model

In the first iteration, we fine-tuned the small checkpoint of the Whisper pre-trained model with the Common Voice dataset in swedish language. This iteration was computation intensive, requiring two full sessions from Colab to complete the model training. The iteration was performed on one notebook as a single monolithic [pipeline](Lab2/base_model/Swedish_fine_tune_whisper.ipynb). The following evaluation was achieved:

* eval/loss: 0.30
* eval/wer: 19.89

There were some factors that may have affected the model score. One is that when resuming the training of the model, the checkpoints from the Whisper-small was used and not the ones that came with the checkpoint e.g. 'checkpoint-2000'. When testing the inference of the model, it did a good job in transcribing some sample audio. (see [log_base](Lab2/base_model/log_base.csv))

## Improving pipeline scalability and model performance

The initial model showed good results, and could be improved by e.g. fine-tuning hyper-parameters or training models with new, better data sources. The following two ways are currently used to train and improve models in the AI/ML community:

- Model-centric approach: Collect any available data and develop a model that is good enough to handle any noise. The data is fixed and the focus lies on iteratively improving the model. 
- Data-centric approach: Focus lines on the quality of the data, thus you try to improve the quality through data preparation methods such that the data is reliable and efficient. With good data, multiple models should be able to perform well.

For improving the model, we chose the *Data-centric approach*. The reason being that training a model is only a small part of the ML lifecyle, while data collection and preparation is a much larger domain. Improving the quality of the data would not only increase the accuracy of the trained model but also multiple other models, in contrast to optimizing a single model to perform well on poorly prepared data with noise.

# Model 2: Improved model (WIP)
In this model, we opted to downsample the dataset for faster training. This model is trained on 20% of the training and validation dataset combined, and evaluated on 20% of the test set. We also included the features 'Accent', 'Gender' and 'upvote_score' as we belive these may have predictive power, and help the model perform better. At this time the ML pipeline was split into the following pipelines:

* [Feature Engineering pipeline](Lab2/new_model/Feature_engineering_Swedish_fine_tune_whisper.ipynb)
* [Training pipeline](Lab2/new_model/Training_pipeline_Swedish_fine_tune_whisper.ipynb)
* [Model inference program](https://huggingface.co/spaces/AbyelT/Swedish-language-transformer) (Uses the base case model)

The following evaluation was achieved:

* eval/loss: WIP
* eval/wer: WIP

Due to time constraints we used the 1000-checkpoint out of 2000 steps, the minimal amount of data and the less time for training the model likely affected the evaluation score of this model. We believe the model could have achieved a better score had it been trained on the full dataset, with longer training steps and using the same features as chosen now. 

## Models
* [Base case model](https://huggingface.co/AbyelT/Whisper-models)
* [Downsampled new model](https://huggingface.co/AbyelT/Whisper-small-better)
