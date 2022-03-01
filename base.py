#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 15:38:53 2021

@author: necva
"""
import utils
from network import Network
import torch
from torchtext.legacy.data import Field,LabelField,BucketIterator,TabularDataset
from tqdm import tqdm
import torch.nn.functional as F

def train_network(network,train_iter,optimizer,loss_fn,epoch_num):
    '''
    train the network using given parameters
    Input:
        network: any Neural Network object 
        train_batch: iterator of training data
        optimizer: optimizer for gradients calculation and updation
        loss_fn: appropriate loss function
        epoch_num = Epoch number so that it can show which epoch number in tqdm Bar
    Returns:
        a tuple of (average_loss,average_accuracy) of floating values for a single epoch
    '''
    epoch_loss = 0
    epoch_acc = 0 
    network.train() 
    
    for batch in tqdm(train_iter,f"Epoch: {epoch_num}"): 
        optimizer.zero_grad() 
        predictions = network(batch.sentences).squeeze(1) 
        loss = loss_fn(predictions,batch.label_encoded.to(torch.long)) 
        pred_classes = F.softmax(predictions, dim=1)
        pred_classes = torch.argmax(pred_classes, dim=1)
        correct_preds = (pred_classes == batch.label_encoded).float()
        accuracy = correct_preds.sum()/len(correct_preds)# it'll be a tensor of shape [1,]
        loss.backward() 
        optimizer.step()
        
        epoch_loss += loss.item() 
        epoch_acc += accuracy.item()
        
        
    return epoch_loss/len(train_iter), epoch_acc/len(train_iter)

def evaluate_network(network,val_test_iter,optimizer,loss_fn):
    '''
    evaluate the network using given parameters
    args:
        network: any Neural Network object 
        val_test_iter: iterator of validation/test data
        optimizer: optimizer for gradients calculation and updation
        loss_fn: appropriate loss function
    out:
        a tuple of (average_loss,average_accuracy) of floating values for the incoming dataset
    '''
    total_loss = 0 
    total_acc = 0
    network.eval()
    
    with torch.no_grad():
        
        for batch in val_test_iter:

            predictions = network(batch.sentences).squeeze(1)
            loss = loss_fn(predictions,batch.label_encoded.to(torch.long))
            pred_classes = torch.argmax(predictions, dim=1)
            correct_preds = (pred_classes == batch.label_encoded).float()
            accuracy = correct_preds.sum()/len(correct_preds)
            total_loss += loss.item() 
            total_acc += accuracy.item()

        return total_loss/len(val_test_iter), total_acc/len(val_test_iter)
    
    
def prepare_dataset_base(parameters, dataset, train):
    """ Prepare dataset
    Input:
        dataset - dataframe 
    Returns:
        train_dataset - BucketIterator train Dataset
        val_dataset - BucketIterator val Dataset
        test_dataset - BucketIteratortest Dataset
        input_size int input size of the model
    """
    
    text_field = Field(tokenize=utils.tokenize)
    label_field = LabelField(dtype=torch.float) 
    # useful for label string to LabelEncoding. Not useful here but doesn't hurt either
    
    fields = [('sentences',text_field),('label_encoded',label_field)] 
    # (column name,field object to use on that column) pair for the dictonary
    
    glove = vocab.Vectors(embedding_path, out_path)
    if train: #prepare train val and est dataset

        if !parameters.dataset: #ıf dataset is not saved
            train_df, val_df, test_df = utils.split_dataset_base(dataset)
            
        train_dataset, val_dataset, test_dataset = TabularDataset.splits(path=parameters.out_path, train='train.csv',validation='val.csv',test='test.csv', 
                                                 format='csv',skip_header=True,fields=fields)
        
        
        
        text_field.build_vocab(train_dataset,max_size=100000,vectors=glove,unk_init=torch.Tensor.zero_) 
        label_field.build_vocab(train_dataset) 
    
        train_iter, val_iter, test_iter = BucketIterator.splits((train_dataset, val_dataset, test_dataset), batch_sizes=(32,128,128),
                                                      sort_key=lambda x: len(x.sentences),
                                                      sort_within_batch=False,
                                                      device=device) # use the cuda device if available
        return train_iter, val_iter, test_iter, input_size
    else: #prepare dataset for tes
        test_dataset = TabularDataset.splits(path=parameters.out_path, test='test.csv', 
                                             format='csv',skip_header=True,fields=fields)
        
        text_field.build_vocab(test_dataset,max_size=100000,vectors=glove,unk_init=torch.Tensor.zero_) 
        label_field.build_vocab(test_dataset)
        
        test_iter = BucketIterator.splits(test_dataset, batch_sizes=32,
                                                      sort_key=lambda x: len(x.sentences),
                                                      sort_within_batch=False,
                                                      device=device) # use the cuda device if available
        input_size = len(text_field.vocab)
        return test_dataset, input_size
    

def test_base_model(parameters, dataset, model, model_path, data):
    """ This function call base model
    Input:
        parameters - Namepace parameters for training
        dataset - dataframe 
        model - string base model
        model_path string loaded pretraine model,
        data dataframe
    """
    
    ptimizer = torch.optim.Adam(network.parameters(), lr=parameters.lr) 
    loss_fn = torch.nn.CrossEntropyLoss() 
    
    test_iter, in_neuron  = prepare_dataset_base(parameters, dataset)
    network = Network(in_neuron, m_type=model)
    network.load_state_dict(torch.load(model_path), strict=False)
    test_loss,test_acc = evaluate_network(network,test_iter,optimizer,loss_fn)
    
def train_base_model(parameters, dataset, model):
    """ This function call pretrained model
    Input:
        parameters - Namepace parameters for training
        dataset - dataframe 
        model - string base model
    Returns:
        tokenizer
        model
    """
    
    num_output = len(set(train_data["label_encoded"])) # number of classes in the dataset
    
    np.random.seed(parameters.seed) 
    torch.manual_seed(parameters.seed)
    torch.cuda.manual_seed(parameters.seed)
    torch.backends.cudnn.deterministic = True  # cuda algorithms
    os.environ['PYTHONHASHSEED'] = str(parameters.seed)
    num_epoch = parameters.num_epochs
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use 'cuda' if available else 'cpu'
    train_iter, val_iter, test_iter, in_neuron  = prepare_dataset_base(parameters, dataset)
    
    network = Network(in_neuron, m_type=model) 
    if torch.cuda.is_available():
        network.cuda() 
    
    optimizer = torch.optim.Adam(network.parameters(),lr=parameters.lr) 
    loss_fn = torch.nn.CrossEntropyLoss() 
    
    for epoch in range(num_epoch):
        train_loss, train_acc = train_network(network,train_iter,optimizer,loss_fn,epoch+1)
        #val_loss,val_acc = evaluate_network(network,val_iter,optimizer,loss_fn)
        tqdm.write(f'''End of Epoch: {epoch+1}  |  Train Loss: {train_loss:.3f}  |  Val Loss: {val_loss:.3f}  |  Train Acc: {train_acc*100:.2f}%  |  Val Acc: {val_acc*100:.2f}%''')
        
    test_loss,test_acc = evaluate_network(network,test_iter,optimizer,loss_fn)