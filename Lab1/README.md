# Lab 1: Serverless ML with Iris Flower Dataset & Titanic Survival Dataset

## Intro
The purpose with this lab is to learn how to setup a serverless ML system with existing tools e.g Modal, Hopsworks and Huggingface Spaces. This lab uses the Iris Flower Dataset and the Titanic dataset, The lab has the following tasks:

1. Build and run a feature pipeline on Modal
2. Build and run a training pipeline on Modal
3. Build and run an inference pipeline with a Gradio UI on Huggingface Spaces

## Tools
- Feature Store: Hopsworks
- Feature & training pipeline: Modal
- Inference pipeline: Hugging Face Spaces 

## Architecture

- Feature Pipeline
  The first step is to process the data, this is done by taking the dataset as input and process it in the feature pipeline. The steps carried out to generate the feature group was:
    - Load the dataset
    - Perform feature engineering - including dropping with few/no predictive power, imputations, binning, casting of datatypes.
    - Upload the schema to Hopsworks.
    - Inserting data into the generated feature group to be used for the training pipeline.
- Training Pipeline
  The training pipeline takes the feature group as input and generates the feature view. The feature view contains the dataset. This dataset is split into train- and test data and we seperate the features and the label to be predicted. The model chosen is the RandomForest Classifier with 100 trees and a depth of 3. Further hyperparameter tuning could have been performed, in this case we went for commonly used ones and since we achieved good accuracy, hence we did not proceed with any further hyperparameter tuning. Furthermore, a confusion matrix is created to showcase the number of true/false negatives and positives. Lastly, the model is uploaded and stored together with the confusion matrix in hopsworks model registry.


- Feature Pipeline Daily
  The feature pipeline daily firstly generates a data entry (a iris flower, or a titanic passenger) with random values for its features. The idea behind the daily feature pipeline is that it would ideally be run once a day on Modal. However there is a difference for the Iris task and the Titanic task. In the case of the titanic passenger, the generated value for the survival label is not dependent on the other features. 

- Batch Inference Pipeline
  The batch-inference pipeline combines and presents a daily prediction with the generated data entry from the daily feature pipeline together with previous predictions

## Task 1 

- Huggingface Interface Links:
  - https://huggingface.co/spaces/AbyelT/Iris
  - https://huggingface.co/spaces/moahof/iris-monitoring

## Task 2

- Huggingface Interface Links:
  - https://huggingface.co/spaces/moahof/Titanic
  - https://huggingface.co/spaces/moahof/titanic-monitoring