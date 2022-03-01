#!/usr/bin/env python3
# -*- coding: utf-8 -*-import torch

class Network(torch.nn.Module):
    '''
    It inherits the functionality of Module class from torch.nn whic includes al the layers, weights, grads setup
    and methods to calculate the same. We just need to put in the required layers and describe the flows as
    which layers comes after which one
    '''
    
    def __init__(self,in_neuron,embedding_dim=50,hidden_size=256,out_neuron=8,m_type='lstm',drop=0.2,**kwargs):
        '''
        Constructor of the class which will instantiate the layers while initialisation.
        
        Input:
            in_neuron: input dimensions of the first layer {int}
            embedding_dim: number of latent features you want to calculate from the input data {int} default=128
            hidden_size: neurons you want to have in your hidden RNN layer {int} default=256
            out_neuron: number of outputs you want to have at the end.{int} default=1
            model: whether to use 'rnn' or 'lstm' {string} 
            drop: proportion of values to dropout from the previous values randomly {float 0-1} default=0.53
            **kwargs: any torch.nn.RNN or torch.nn.LSTM args given m_type='rnn' or'lstm' {dict}
        Returns: 
            A tensor of shape {batch,out_neuron} as output 
        '''
        super(Network,self).__init__() 
        self.m_type = m_type
        
        self.embedding = torch.nn.Embedding(in_neuron,embedding_dim) # embedding layer is always the first layer
        if self.m_type == "bilstm":
            self.bilstm = torch.nn.LSTM(embedding_dim,hidden_size,bidirectional=True, **kwargs)
        elif self.m_type == 'lstm':
        # whether to use the LSTM type model or the RNN type model. It'll use only 1 in forward()
            self.lstm = torch.nn.LSTM(embedding_dim,hidden_size,**kwargs)
        else:
            self.rnn = torch.nn.RNN(embedding_dim,hidden_size,**kwargs) 
        
        self.dropout = torch.nn.Dropout(drop) # drop the values by random which comes from previous layer
        if self.mtype == "bilstm":
            self.dense = torch.nn.Linear(hidden_size*2,out_neuron) # last fully connected layer
        else:
            self.dense = torch.nn.Linear(hidden_size*2,out_neuron) # last fully connected layer
    
    def forward(self,t):
        '''
        Activate the forward propagation of a batch at a time to transform the input bath of tensors through
        the different layers to get an out which then will be compared to original label for computing loss.
        Input:
            t: tensors in the form of a batch {torch.tensor}
        Returns:
            output of the network
        '''
        embedding_t = self.embedding(t)
        
        drop_emb = self.dropout(embedding_t)
        
        if self.m_type == "bilstm":
            out, (hidden_state,_) = self.bilstm(drop_emb)
            hidden_state = torch.cat((hidden_state[0,:,:],hidden_state[1,:,:]), dim=1)
        elif self.m_type == 'lstm':
            out, (hidden_state,_) = self.lstm(drop_emb)
        else:
            out, hidden_state = self.rnn(drop_emb)
            #  shape of rnn_out = (seq_len, batch, num_directions * hidden_size)
       
        hidden_squeezed = hidden_state.squeeze(0) 
        
        return self.dense(hidden_squeezed)