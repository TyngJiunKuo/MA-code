import torch
import torchtext
from torchtext import data
from torch.utils.data import Dataset
from torchtext import datasets
import random
import networks
import os
# import numpy as np
import spacy
#from torchtext.vocab import Vectors
import json
import numpy as np

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imdb_path = '/home/kuo/code/data/aclImdb/'  
sst_path = '/home/kuo/code/data/sst/trainDevTestTrees_PTB.zip'  
spacy_en = spacy.load('en_core_web_sm')
#vectors = Vectors(name='glove.6B.100d.txt', cache='/home/kuo/code/.vector_cache/')


def tokenize_en(sentence):
    return [tok.text for tok in spacy_en.tokenizer(sentence)]
    
class DataBinary(Dataset):   # prepare bert style instances.
    def __init__(self, instances, tokenizer,  maxlen):
        self.maxlen = maxlen
        self.instances = [[self.pad(sent[0], tokenizer, maxlen), sent[1]] for sent in instances]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        x = self.instances[idx][0]
        y = self.instances[idx][1]
        return x, y

    def pad(self, instance, tokenizer, maxlen):    # pad value is  0 in bert tokenizer
        padded = np.zeros((maxlen, ), dtype=np.int64)
        if len(instance) > maxlen - 2:
            instance = [tokenizer.cls_token] + instance[:maxlen -2] + [tokenizer.sep_token]
            padded[:] = tokenizer.convert_tokens_to_ids(instance)
        else:
            instance = [tokenizer.cls_token] + instance + [tokenizer.sep_token]
            padded[:len(instance)] = tokenizer.convert_tokens_to_ids(instance)

        return padded

    
#label2id_binary = {"0": 0, "1": 1}   # for SST y label.
#data_root = '/home/lyu/robustness/Datasets/'
#data_root ='/home/kuo/code/data/sst'


def load_splits_json(which_data):
    def load_json(file):
        with open(file, 'r')as f:
            data = json.load(f)
        return data

    #print('Loading {} dataset...'.format(which_data))
    directory = '/home/kuo/code/data/sst'
    #directory = os.path.join(data_root, which_data+'data')
    #json_file = os.path.join(directory, which_data+'_input.json')
    train_ids = os.path.join(directory, 'train.json')
    dev_ids = os.path.join(directory, 'dev.json')
    test_ids = os.path.join(directory, 'test.json')
    #All_samples = load_json(json_file)
    train_data = load_json(train_ids)
    dev_data = load_json(dev_ids)
    test_data = load_json(test_ids)
    train_samples = [(train_data[i]['en_defs'], train_data[i]['label']) for i in range(len(train_data))]
    dev_samples = [(dev_data[i]['en_defs'], dev_data[i]['label']) for i in range(len(dev_data))]
    test_samples = [(test_data[i]['en_defs'], test_data[i]['label']) for i in range(len(test_data))]
    return train_samples, dev_samples, test_samples

    
def load_data_for_bert(which_data, tokenizer):
    label2id_binary = {"0": 0, "1": 1}   
    if which_data == 'sst':
        label2id = label2id_binary
        MAX_LENGTH = 30
    else:
        raise ValueError('Datasets:  SST.')
    #print('Loading {} data for bert model...'.format(which_data))
    train_samples, dev_samples, test_samples = load_splits_json(which_data)    # load SST data, which is json object of list of (text, label) tuple.
    train_ids = [ [tokenizer.tokenize(sample[0]), label2id[sample[1]]] for sample in train_samples]
    dev_ids = [ [tokenizer.tokenize(sample[0]), label2id[sample[1]]] for sample in dev_samples]
    test_ids = [ [tokenizer.tokenize(sample[0]), label2id[sample[1]]] for sample in test_samples]

    train_data = DataBinary(train_ids, tokenizer, MAX_LENGTH)
    dev_data = DataBinary(dev_ids, tokenizer, MAX_LENGTH)
    test_data = DataBinary(test_ids, tokenizer, MAX_LENGTH)
    return train_data, dev_data, test_data


def load_imdb(MAX_VOCAB_SIZE = 50000, SEED = 1234, BATCH_SIZE=64, only_vocab=False):
    TEXT = data.Field(tokenize = tokenize_en, batch_first = True, include_lengths = True)
    LABEL = data.LabelField(dtype = torch.long)

    #SST = datasets.SST(sst_path, TEXT, LABEL)
    #train_data, test_data = SST.splits(TEXT, LABEL, path=sst_path)

    IMDB = datasets.IMDB(imdb_path, TEXT, LABEL)
    train_data, test_data = IMDB.splits(TEXT, LABEL, path=imdb_path)
    #train_data, test_data = datasets.SST.splits(TEXT, LABEL)
    
    train_data, valid_data = train_data.split(random_state = random.seed(SEED))
    
    TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 #vectors = vectors,  
                 unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(train_data)
    
    if only_vocab:
        return TEXT, LABEL
    else:
        train_iterator, valid_iterator, test_iterator = iter_data(train_data, valid_data, test_data, BATCH_SIZE)
        return TEXT, LABEL, train_iterator, valid_iterator, test_iterator
    

def load_sst(MAX_VOCAB_SIZE=50000, SEED=1234, BATCH_SIZE=32, only_vocab=False):
    TEXT = data.Field(tokenize = tokenize_en, batch_first = True, include_lengths = True)#, lower = True)
    LABEL = data.LabelField(dtype = torch.long)

    #SST = datasets.SST(sst_path, TEXT, LABEL)
    #train_data, valid_data, test_data = SST.splits(TEXT, LABEL, path = sst_path, train_subtrees=True, filter_pred=lambda ex: ex.label != 'neutral')
    #train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL, train_subtrees=True, filter_pred=lambda ex: ex.label != 'neutral')
    #train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL, train_subtrees=True)

    #train_data, valid_data = train_data.split(random_state = random.seed(SEED))
    train_data, valid_data, test_data = data.TabularDataset.splits(path='/home/kuo/code/data/sst/',
                                       train='test.tsv',
                                       validation='dev.tsv',
                                       test = 'test.tsv',
                                       format='tsv',
                                       skip_header=False,
                                       fields=[('label', LABEL), ('text', TEXT)])

    TEXT.build_vocab(train_data,
            max_size = MAX_VOCAB_SIZE,
            vectors = 'glove.6B.100d',
            unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(train_data)

    if only_vocab:
        return TEXT, LABEL
    else:
        train_iterator, valid_iterator, test_iterator = iter_data(train_data, valid_data, test_data, BATCH_SIZE)
        return TEXT, LABEL, train_iterator, valid_iterator, test_iterator


def iter_data(train, valid, test, BATCH_SIZE):

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train, valid, test), 
    batch_size = BATCH_SIZE,
    #sort_within_batch = True,
    sort_within_batch = False, #for TabularDataset it need to set as False otherwise cause error because it will start to compare the batch, but '<' was not supported
    sort = False,
    device = device)
    
    return train_iterator, valid_iterator, test_iterator


def random_imdb_test_sample(seed = 100, length=1000):       # original_length = 1000
    TEXT = data.Field(tokenize=tokenize_en, batch_first=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.long)

    IMDB = datasets.IMDB(imdb_path, TEXT, LABEL)
    _, test_data = IMDB.splits(TEXT, LABEL, path=imdb_path)
    #_, test_data = datasets.SST.splits(TEXT, LABEL)
    random.seed(seed)
    index = random.sample(range(len(test_data)), length)
    test_sample = [(' '.join(test_data[i].text), test_data[i].label) for i in index]
    return test_sample

def random_sst_test_sample(seed = 100, length=100):       # original_length = 1000
    TEXT = data.Field(tokenize=tokenize_en, batch_first=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.long)

    #SST = datasets.SST(sst_path, TEXT, LABEL)
    #_, _, test_data = SST.splits(TEXT, LABEL, path=sst_path)
    _, _, test_data = datasets.SST.splits(TEXT, LABEL)
    random.seed(seed)
    index = random.sample(range(len(test_data)), length)
    test_sample = [(' '.join(test_data[i].text), test_data[i].label) for i in index]
    return test_sample


def tokens_to_sents(tokens):
    return ' '.join(tokens)
