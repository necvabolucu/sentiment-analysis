#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretrained models
"""

from dataset import Dataset
import utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import EarlyStoppingCallback


def bert_model(output_label):
    """ Define bert pretrained tokenizer and model
    Input:
        output_label - int the number of classes in the dataset
    Returns:
        tokenizer
        model
    """
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=output_label)
    
    return tokenizer, model

def test_bert_model(model_path, dataset):
    """ Test with bert pretrained model
    Input:
        model_path - path of saved pretrained model
    Returns:
        predictions list predictions of test dataset
    """
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=8) 
    test_trainer = Trainer(model)
    raw_pred, _, _ = test_trainer.predict(dataset) 
    predictions = np.argmax(raw_pred, axis=1)
    
    return predictions
    
    
def distilbert_model(output_label):
    """ Define distilbert pretrained tokenizer and model
    Input:
        output_label - int the number of classes in the dataset
    Returns:
        tokenizer
        model
    """
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=output_label)
    
    return tokenizer, model

def test_distilbert_model(model_path, dataset):
    """ Test with distilbert pretrained model
    Input:
        model_path - path of saved pretrained model
    Returns:
        predictions list predictions of test dataset
    """
    model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=8) 
    test_trainer = Trainer(model)
    raw_pred, _, _ = test_trainer.predict(dataset) 
    predictions = np.argmax(raw_pred, axis=1)
    
    return predictions

def alberta_model(output_label):
    """ Define alberta pretrained tokenizer and model
    Input:
        output_label - int the number of classes in the dataset
    Returns:
        tokenizer
        model
    """
    model_name = "albert-base-v2"
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=output_label)

    return tokenizer, model

def test_alberta_model(model, dataset):
    """ Test with alberta pretrained model
    Input:
        model_path - path of saved pretrained model
    Returns:
        predictions list predictions of test dataset
    """
    model = AlbertForSequenceClassification.from_pretrained(model_path, num_labels=8) 
    test_trainer = Trainer(model)
    raw_pred, _, _ = test_trainer.predict(dataset) 
    predictions = np.argmax(raw_pred, axis=1)
    
    return predictions

def gpt2_model(output_label):
    """ Define GPT2 pretrained tokenizer and model
    Input:
        output_label - int the number of classes in the dataset
    Returns:
        tokenizer
        model
    """
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=output_label)
    

    return tokenizer, model

def test_gpt2_model(model_path, dataset):
    """ Test with alberta pretrained model
    Input:
        model_path - path of saved pretrained model
    Returns:
        predictions list predictions of test dataset
    """
    model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=8) 
    test_trainer = Trainer(model)
    raw_pred, _, _ = test_trainer.predict(dataset) 
    predictions = np.argmax(raw_pred, axis=1)
    
    return predictions

def prepare_dataset_pretrained(tokenizer, dataset):
    """ Prepare dataset
    Input:
        tokenizer - pretrained tokenizer
        dataset - dataframe 
    Returns:
        train_dataset - torch.utils.data.Dataset train Dataset
        val_dataset - torch.utils.data.Dataset val Dataset
        test_dataset - torch.utils.data.Dataset test Dataset
        y_test - list gold labels for the test data
    """
    X_train, y_train, X_val, y_val, X_test, y_test = utils.split_dataset_pretrained(dataset)
    
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)      
    
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)
    test_dataset = Dataset(X_test_tokenized, y_test)
    
    return train_dataset, val_dataset, test_dataset, y_test


def test(trainer, test_dataset, y_test):
    """ This function calculate accuracy of the test
    Input:
        trainer - Trainew
        test_dataset - dataset
        y_test list gold labels of test dataset
    """
    
    raw_pred, _, _ = trainer.predict(test_dataset)
    
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    
    y_true = y_test
    y =zip(y_pred, y_true)
    print(utils.compute_metrics(y))    
 
def train(parameters, tokenizer, model, train_dataset, val_dataset, output_path):
    """ This function call pretrained model
    Input:
        parameters - Namepace parameters for training
        tokenizer - pretrained tokenizer 
        model - pretrained model
        train_dataset - dataset
        val_dataset - dataset
    Returns:
        trainer
    """

    args = TrainingArguments(
    output_dir = output_path,
    evaluation_strategy = 'steps',
    eval_steps = 500,
    per_device_train_batch_size = parameters.batch_size,
    per_device_eval_batch_size = parameters.batch_size,
    num_train_epochs = parameters.num_epochs,
    seed = parameters.seed,
    load_best_model_at_end = True,)   
    
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=utils.compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],)
    
    trainer.train()
    
    return trainer

def test_pretrained_model(model_path, data):
    """ This function call pretrained model
    Input:
        model_path string loaded pretraine model,
        data dataframe
    Returns:
        y_pred predictions of the test_data
    """
    X_test_tokenized = tokenizer(list(data["sentences"].values), padding=True, truncation=True, max_length=512)
    test_dataset = Dataset(X_test_tokenized, list(data["label_encoded"].values))
    
    if model == 'bert':
        predictions = test_bert_model(model_path, test_dataset)
    elif model == 'alberta':
        predictions = test_alberta_model(model_path, test_dataset)
    elif model == 'distilbert':
        predictions = test_distilbert_model(model_path, test_dataset)
    elif model == 'gpt2':
        predictions = test_gpt2_model(model_path, test_dataset)
    else:
        print('model is not defined')
        
    y_true = list(data["label_encoded"].values)
    y = predictions, y_true
    print(utils.compute_metrics(y))   

    

def train_pretrained_model(parameters, dataset, model):
    """ This function call pretrained model
    Input:
        parameters - Namepace parameters for training
        dataset - dataframe 
        model - string pretrained model
    Returns:
        tokenizer
        model
    """
    
    num_output = len(set(dataset["label_encoded"])) # number of classes in the dataset
    
    if model == 'bert':
        tokenizer, model = bert_model(num_output)
    elif model == 'alberta':
        tokenizer, model = alberta_model(num_output)
    elif model == 'distilbert':
        tokenizer, model = distilbert_model(num_output)
    elif model == 'gpt2':
        tokenizer, model = gpt2_model(num_output)
    else:
        print('model is not defined')
    
    train_dataset, val_dataset, test_dataset, y_test = prepare_dataset_pretrained(tokenizer, dataset)
    trainer = train(parameters, tokenizer, model, train_dataset, val_dataset, parameters.model_path)
    
    test(trainer, test_dataset, y_test)
    
   