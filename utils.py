#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils
"""
# Libraries
import pandas as pd
import numpy as np
import json
from argparse import Namespace
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

seed = 1234
def common(config_filepath):
    """ Read config file
    Input:
        config_filepath - string the path of the config
    Returns:
        conf Namespace parameters for the training 
    """
    with open(config_filepath, 'r') as config_file:
        conf = json.load(config_file, object_hook=lambda d: Namespace(**d))
        
    return conf

def read_dataset(file_path):
    """ Read dataset
    Input:
        file_path - string the path of the dataset
    Returns:
        train dataframe 
    """
    train_data = pd.read_excel('L2400.xlsx', 'Sheet1')
    
    ''' Should/Must statement
        Should/must statement
        should/must statement labels are 
        converted to Should/Must statement
        
        personalizing is converted to Personalizing''' 
    
    
    train_data.loc[(train_data['label'] == 'should/must statement') | (train_data['label'] == 'Should/must statement')] = 'Should/Must statement' 
    train_data.loc[train_data['label'] == 'personalizing'] = 'Personalizing' 
    
    #Label encoding 
    
    le = LabelEncoder()
    train_data["label_encoded"] = le.fit_transform(train_data["label"]) 
    
    return train_data

def split_dataset_base(dataframe):
    """ Split dataset into train, val and test
    Input:
        dataframe - dataframe dataset
    Returns:
        train_df dataframe train dataframe
        val_df dataframe val dataframe
        test_df dataframe test dataframe
    """   
    # split train dataset into train, validation and test sets
    df, test_df = train_test_split(dataframe,random_state=seed,test_size=0.2, stratify=dataframe["label_encoded"])
    
    train_df, val_df = train_test_split(df,random_state=seed,test_size=0.2, stratify=df["label_encoded"])
    
    save_files(train_df, val_df, test_df)
    
    return train_df, val_df, test_df
    
def save_files(out_path, train_df, val_df, test_df):
    """ Save splittted dataset into folder
    Input:
        out_path string path for saving the files
        train_df dataframe train dataframe
        val_df dataframe val dataframe
        test_df dataframe test dataframe
    """  
    train_df.to_csv(out_path+'train.csv',index=False)
    val_df.to_csv(out_path+'val.csv',index=False)
    test_df.to_csv(out_path+'test.csv',index=False) 
   
def tokenize(s): 
    """ Split text
    Input:
        s string text to split
    Returns:
        string splittex text
    """  
    return s.split(' ')
    
def split_dataset_pretrained(dataframe):
    """ Split dataset into train, val and test
    Input:
        dataframe - dataframe dataset
    Returns:
        X_train list train sentences
        y_train list label of train dataset
        X_val list val sentences
        y_val list label of val dataset
        X_test list test sentences
        y_test list label of test dataset
    """
    X_train, temp_text, y_train, temp_labels = train_test_split(list(dataframe["sentences"].values), list(dataframe["label_encoded"].values), 
                                                                    random_state=seed, 
                                                                    test_size=0.2, 
                                                                    stratify=list(dataframe["label_encoded"].values))


    X_val, X_test, y_val, y_test = train_test_split(temp_text, temp_labels, 
                                                                random_state=seed, 
                                                                test_size=0.4, 
                                                                stratify=temp_labels)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def compute_metrics(p):
    """Compute metrics for evaluation
    p Lists prediction and gold labels for evaluation
    Reurns:
        eval_scores dictionary evaluation scores
    """
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='micro')
    precision = precision_score(y_true=labels, y_pred=pred, average='micro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='micro')

    eval_scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    return eval_scores