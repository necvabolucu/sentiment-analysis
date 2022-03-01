# Sentiment fine-grained classifications-8

This project contains the code that was used to obtain the results of the deep learning models.


## Introduction
The state-of-the-art pretrained language model BERT (Bidirectional Encoder Representations from Transformers) has achieved remarkable results in many natural language understanding tasks. In general, the industry-wide adoption of transformer architectures (BERT, DistlERT, etc.) marked a sharp deviation from the conventional encoder-decoder architectures in sequence-to-sequence tasks such as machine translation. Following this, we utilized the pretrained weights of these language representation models, and *fine-tuning* them to suitsentiment classification task.

We also adopt base deep learning models to compare results with pretrained language models.

## Requirements

```
python >= 3.6.0
numpy == 1.19.5 
pandas == 1.3.3   
torch == 1.9.1  
ucca == 1.0.127
transformers == 4.11.3 
sckit-learn == 1.0
sentencepiece == 0.1.96  
torchtext == 0.10.1 
tqdm == 4.62.3 
```


## Usage
Experiments for various configuration can be using the ```run.py```. First of all, install the python packages (preferably in a clean virtualenv): ```pip install -r requirements.txt```


```sh
$ python3 run.py [OPTIONS]
usage: run.py --config config.json

```

# Experiments 
## Comparison & Analysis of Results
### Baseline Models
#### RNN 
Accuracy : 59.50
#### LSTM
Accurcay : 64.47
#### BILSTM
Accurcay : 66.34

### Pretrained Models
#### BERT
##### base
Accuracy    |   Precision   |   Recall    | F1
75.75   |   75.52   |   75.52   | 75.52
##### large
Accuracy    |   Precision   |   Recall    | F1
77.60  |   77.60  |   77.60   | 77.60
#### DistilBERT
Accuracy    |   Precision   |   Recall    | F1
74.47  |   74.47  |   74.47   | 74.47
#### AlBERTa
#### GPT2
Accuracy    |   Precision   |   Recall    | F1
73.43 |   73.43  |   73.43   | 73.43
