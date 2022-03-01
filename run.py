# -*- coding: utf-8 -*-
# Sentiment fine-grained classifications-8

#Libraries
import sys
import argparse
import utils
import pretrained 
def main(args):
    print('Sentiment fine-grained classifications-8 project')
    parameters = utils.common(args.config)

    if parameters.train:
        print('training')
        dataset = utils.read_dataset(parameters.data)
        if parameters.model == 'pretrained':  # BERT, Alberta, DistilBert, GPT2
            pretrained.train_pretrained_model(parameters, dataset, parameters.submodel)
        elif parameters.model == 'base': # LSTM, BILSTM, RNN:
            base.train_base_model(parameters, dataset, parameters.submodel)
        else:
            print('The model is not valid')
    else:
        print('Test')
        dataset = utils.read_dataset(parameters.test_data)
        if parameters.model == 'pretrained':  # BERT, Alberta, DistilBert, GPT2
            pretrained.test_pretrained_model(parameters, dataset, parameters.submodel)
        elif parameters.model == 'base': # LSTM, BILSTM, RNN:
            base.train_base_model(parameters, dataset, parameters.submodel)
        else:
            print('The model is not valid')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Sentiment fine-grained classifications-8')
    parser.add_argument('--config', help='path of the config file')
    args = parser.parse_args()
    main(args)